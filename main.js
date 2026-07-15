// Orbitrap 3D Physics Simulator - Core engine & UI

const R_m = 25.0;       // Characteristic radius of Orbitrap (mm)
const R_in0 = 15.0;     // Spindle radius at z=0 (mm)
const R_out0 = 40.0;    // Outer barrel radius at z=0 (mm)
const L = 50.0;         // Half-length of the Orbitrap (mm)
const V_accel = 160.0;  

// Time and playback state
const baseDt = 0.02;
let simSpeed = 2.0;
let isPlaying = true;

// Voltage ramping state
let voltageRamping = true;
let V_init = 200.0;
let V_final = 2000.0;
let t_ramp_time = 3.0;
let currentVoltage = 200.0;
let timeElapsed = 0.0;

// Vacuum quality & damping
let vacuumQuality = 99.998;
let dampingCoef = 0.0;

// Electrode display options
let cutawayAngle = Math.PI;
let outerOpacity = 40;
let innerOpacity = 90;
let showGrid = true;

// Active ions array
let ions = [];
let nextIonId = 1;
let physicsStepCount = 0;
let inspectedIonIndex = 0;

// Signal buffer for FFT and Oscilloscope
// Retain a complete 16.777 ms acquisition rather than a short rolling trace.
// At m/z 300 this is long enough to resolve peaks separated by about 0.1 m/z.
const transientSize = 262144;
let rawSignal = new Float32Array(transientSize);
let signalPtr = 0;
let signalCount = 0;
const detectorSampleInterval = baseDt * 16 / 5;
const detectorSettlingTime = 2.0;
let detectorSampleAccumulator = 0;
let detectorAcquiring = false;
let detectorNoise = 5.0;
let signalAbsMax = 0.01;

// Interactive mass-spectrum view state
const fullSpectrumRange = { min: 150.0, max: 650.0 };
let spectrumView = { ...fullSpectrumRange };
let spectrumPeakHitTargets = [];
let selectedSpectrumPeak = null;

// FFT plans and results are cached because the acquisition can be large.
const fftPlans = new Map();
let cachedFftMag = new Float32Array(0);
let cachedFftSize = 0;
let cachedFftSignalCount = -1;
let cachedFftUpdatedAt = 0;
const spectrumRefreshIntervalMs = 250;
let spectrumRemoveBaseline = false;
let spectrumHannWindow = false;
let spectrumZeroFill = false;

function getFFTPlan(fftSize) {
    const cachedPlan = fftPlans.get(fftSize);
    if (cachedPlan) return cachedPlan;

    const revTable = new Int32Array(fftSize);
    const cosTable = new Float32Array(fftSize / 2);
    const sinTable = new Float32Array(fftSize / 2);
    let limit = 1;
    let bit = fftSize >> 1;
    while (limit < fftSize) {
        for (let i = 0; i < limit; i++) {
            revTable[i + limit] = revTable[i] + bit;
        }
        limit <<= 1;
        bit >>= 1;
    }
    for (let i = 0; i < fftSize / 2; i++) {
        let angle = -2 * Math.PI * i / fftSize;
        cosTable[i] = Math.cos(angle);
        sinTable[i] = Math.sin(angle);
    }

    const plan = { revTable, cosTable, sinTable };
    fftPlans.set(fftSize, plan);
    return plan;
}

function performFFT(re, im) {
    const fftSize = re.length;
    const { revTable, cosTable, sinTable } = getFFTPlan(fftSize);
    for (let i = 0; i < fftSize; i++) {
        let j = revTable[i];
        if (i < j) {
            let temp = re[i]; re[i] = re[j]; re[j] = temp;
            temp = im[i]; im[i] = im[j]; im[j] = temp;
        }
    }
    for (let size = 2; size <= fftSize; size <<= 1) {
        let halfSize = size >> 1;
        let tabStep = fftSize / size;
        for (let i = 0; i < fftSize; i += size) {
            for (let j = 0; j < halfSize; j++) {
                let k = i + j;
                let l = k + halfSize;
                let t_cos = cosTable[j * tabStep];
                let t_sin = sinTable[j * tabStep];
                
                let tRe = re[l] * t_cos - im[l] * t_sin;
                let tIm = re[l] * t_sin + im[l] * t_cos;
                
                re[l] = re[k] - tRe;
                im[l] = im[k] - tIm;
                re[k] += tRe;
                im[k] += tIm;
            }
        }
    }
}

function invalidateSpectrumCache() {
    cachedFftMag = new Float32Array(0);
    cachedFftSize = 0;
    cachedFftSignalCount = -1;
    cachedFftUpdatedAt = 0;
}

function refreshSpectrumFFT(validSignalCount, force = false) {
    if (validSignalCount < 128) return;

    const now = performance.now();
    const hasCurrentTransform = cachedFftSignalCount === validSignalCount;
    const newSamples = validSignalCount - cachedFftSignalCount;
    const minimumNewSamples = Math.max(1, Math.floor(Math.max(cachedFftSignalCount, 0) * 0.005));
    const acquisitionJustCompleted = validSignalCount === transientSize;
    if (hasCurrentTransform || (!force && cachedFftSize > 0 && !acquisitionJustCompleted && (
        now - cachedFftUpdatedAt < spectrumRefreshIntervalMs || newSamples < minimumNewSamples
    ))) {
        return;
    }

    // Grow the transform with the acquisition. Padding only to the next power
    // of two keeps updates responsive without pretending to add resolution.
    const baseFFTSize = Math.min(
        transientSize,
        2 ** Math.ceil(Math.log2(validSignalCount))
    );
    const fftSize = baseFFTSize * (spectrumZeroFill ? 2 : 1);
    const fftRe = new Float32Array(fftSize);
    const fftIm = new Float32Array(fftSize);

    let baseline = 0;
    if (spectrumRemoveBaseline) {
        for (let i = 0; i < validSignalCount; i++) baseline += rawSignal[i];
        baseline /= validSignalCount;
    }
    for (let i = 0; i < validSignalCount; i++) {
        const windowScale = spectrumHannWindow && validSignalCount > 1
            ? 0.5 * (1 - Math.cos((2 * Math.PI * i) / (validSignalCount - 1)))
            : 1;
        fftRe[i] = (rawSignal[i] - baseline) * windowScale;
    }
    performFFT(fftRe, fftIm);

    const fftMag = new Float32Array(fftSize / 2);
    for (let i = 0; i < fftMag.length; i++) {
        fftMag[i] = Math.hypot(fftRe[i], fftIm[i]);
    }
    cachedFftMag = fftMag;
    cachedFftSize = fftSize;
    cachedFftSignalCount = validSignalCount;
    cachedFftUpdatedAt = now;
}

// Electrode profile precalculations
let innerProfile = [];
let outerProfile = [];
let fullInnerProfile = [];
let fullOuterProfile = [];

function generateElectrodeProfiles() {
    innerProfile = [];
    outerProfile = [];
    fullInnerProfile = [];
    fullOuterProfile = [];

    // Spindle profile
    const r_min = 0.15;
    for (let r = R_in0; r >= r_min; r -= 0.05) {
        let arg = 0.5 * (Math.pow(r / R_m, 2) - Math.pow(R_in0 / R_m, 2)) - Math.log(r / R_in0);
        if (arg >= 0) {
            let z = R_m * Math.sqrt(arg);
            if (z <= L) {
                innerProfile.push({ z: z, r: r });
            }
        }
    }
    innerProfile.sort((a, b) => a.z - b.z);

    // Outer barrel profile
    const r_max = 90.0;
    for (let r = R_out0; r <= r_max; r += 0.05) {
        let arg = 0.5 * (Math.pow(r / R_m, 2) - Math.pow(R_out0 / R_m, 2)) - Math.log(r / R_out0);
        if (arg >= 0) {
            let z = R_m * Math.sqrt(arg);
            if (z <= L) {
                outerProfile.push({ z: z, r: r });
            }
        }
    }
    outerProfile.sort((a, b) => a.z - b.z);

    // Mirrored profiles for 3D rendering
    for (let i = innerProfile.length - 1; i >= 0; i--) {
        fullInnerProfile.push({ z: -innerProfile[i].z, r: innerProfile[i].r });
    }
    for (let i = 0; i < innerProfile.length; i++) {
        fullInnerProfile.push({ z: innerProfile[i].z, r: innerProfile[i].r });
    }

    for (let i = outerProfile.length - 1; i >= 0; i--) {
        fullOuterProfile.push({ z: -outerProfile[i].z, r: outerProfile[i].r });
    }
    for (let i = 0; i < outerProfile.length; i++) {
        fullOuterProfile.push({ z: outerProfile[i].z, r: outerProfile[i].r });
    }
}

function getElectrodeRadius(profile, z) {
    let absZ = Math.abs(z);
    if (absZ > L) return null;
    
    // Linear interpolation
    for (let i = 0; i < profile.length - 1; i++) {
        let p1 = profile[i];
        let p2 = profile[i+1];
        if (p1.z <= absZ && absZ <= p2.z) {
            let t = (absZ - p1.z) / (p2.z - p1.z);
            return p1.r + t * (p2.r - p1.r);
        }
    }
    if (profile.length > 0) return profile[profile.length - 1].r;
    return null;
}

// Physics Engine (Ion Motion Integration)
function getAcceleration(x, y, z, q, m, kVal) {
    let r2 = x * x + y * y;
    let r = Math.sqrt(r2);
    
    if (r < 0.1) {
        return { x: 0, y: 0, z: 0 };
    }
    
    // Orbitrap equations of motion
    let coefRad = (q * kVal) / (2 * m);
    let ax = coefRad * x * ((R_m * R_m) / r2 - 1);
    let ay = coefRad * y * ((R_m * R_m) / r2 - 1);
    let az = -((q * kVal) / m) * z;
    
    return { x: ax, y: ay, z: az };
}

function updatePhysics(dt) {
    physicsStepCount++;
    let kVal = currentVoltage;

    // Calculate damping from vacuum quality
    dampingCoef = (100.0 - vacuumQuality) * 0.0005;

    let totalActiveIons = 0;
    let currentSignalValue = 0.0;

    for (let ion of ions) {
        if (!ion.active) continue;
        totalActiveIons++;

        // Verlet integration
        let a = getAcceleration(ion.x, ion.y, ion.z, ion.q, ion.m, kVal);
        
        let nextX = ion.x + ion.vx * dt + 0.5 * a.x * dt * dt;
        let nextY = ion.y + ion.vy * dt + 0.5 * a.y * dt * dt;
        let nextZ = ion.z + ion.vz * dt + 0.5 * a.z * dt * dt;
        
        let nextA = getAcceleration(nextX, nextY, nextZ, ion.q, ion.m, kVal);
        let dampingFactor = 1.0 - dampingCoef * dt;
        
        ion.vx = (ion.vx + 0.5 * (a.x + nextA.x) * dt) * dampingFactor;
        ion.vy = (ion.vy + 0.5 * (a.y + nextA.y) * dt) * dampingFactor;
        ion.vz = (ion.vz + 0.5 * (a.z + nextA.z) * dt) * dampingFactor;
        
        ion.x = nextX;
        ion.y = nextY;
        ion.z = nextZ;

        // Trace history for 3D path
        ion.history.push({ x: ion.x, y: ion.y, z: ion.z });
        if (ion.history.length > 350) {
            ion.history.shift();
        }

        // Downsample coordinates for 2D plots
        if (physicsStepCount % 8 === 0) {
            ion.tSeriesX.push(ion.x);
            ion.tSeriesY.push(ion.y);
            ion.tSeriesZ.push(ion.z);
            if (ion.tSeriesX.length > 500) {
                ion.tSeriesX.shift();
                ion.tSeriesY.shift();
                ion.tSeriesZ.shift();
            }
        }

        if (!ion.hasLeftInjector && ion.y < -2.0) {
            ion.hasLeftInjector = true;
        }

        // Collision checking
        let r = Math.sqrt(ion.x * ion.x + ion.y * ion.y);
        let absZ = Math.abs(ion.z);

        if (absZ >= L) {
            ion.active = false;
            ion.collisionStatus = "escaped";
        } else if (ion.hasLeftInjector && Math.abs(ion.z - ion.zInject) < 1.0 && Math.abs(ion.y) < 1.0 && ion.x >= (ion.rInject - 1.0)) {
            ion.active = false;
            ion.collisionStatus = "injector";
        } else {
            let r_in = getElectrodeRadius(innerProfile, absZ);
            let r_out = getElectrodeRadius(outerProfile, absZ);

            if (r_in !== null && r <= r_in) {
                ion.active = false;
                ion.collisionStatus = "spindle";
            } else if (r_out !== null && r >= r_out) {
                ion.active = false;
                ion.collisionStatus = "barrel";
            }
        }

        // Image current detection
        if (ion.active) {
            currentSignalValue += ion.q * ion.vz;
        }
    }

    // Add noise
    let noiseAmp = (detectorNoise / 100.0) * 8.0; 
    currentSignalValue += (Math.random() - 0.5) * noiseAmp;

    // Start detector acquisition after settling time
    const acquisitionStart = (voltageRamping ? t_ramp_time : 0) + detectorSettlingTime;
    if (!detectorAcquiring && signalPtr < transientSize && timeElapsed >= acquisitionStart) {
        detectorAcquiring = true;
        detectorSampleAccumulator = 0;
    }

    if (detectorAcquiring) {
        detectorSampleAccumulator += dt;
        while (detectorSampleAccumulator >= detectorSampleInterval && signalPtr < transientSize) {
            detectorSampleAccumulator -= detectorSampleInterval;
            rawSignal[signalPtr] = currentSignalValue;
            signalAbsMax = Math.max(signalAbsMax, Math.abs(currentSignalValue));
            signalPtr++;
            signalCount = signalPtr;
        }
        if (signalPtr >= transientSize) {
            detectorAcquiring = false;
        }
    }

    // Voltage ramp
    if (voltageRamping) {
        if (timeElapsed < t_ramp_time) {
            currentVoltage = V_init + (V_final - V_init) * (timeElapsed / t_ramp_time);
        } else {
            currentVoltage = V_final;
        }
    } else {
        currentVoltage = V_final;
    }

    timeElapsed += dt;
    document.getElementById("stat-voltage").innerText = `${Math.round(currentVoltage)} V`;
    document.getElementById("stat-trapped").innerText = `${totalActiveIons} / ${ions.length}`;
    document.getElementById("stat-time").innerText = `${timeElapsed.toFixed(2)} \u03BCs`;
}

function advanceSimulationFrame() {
    // Keep manual and automatic advancement numerically identical. A single
    // UI step represents one rendered frame, including its stability substeps.
    const stepsPerFrame = Math.max(5, Math.round(5 * simSpeed));
    const dt = (baseDt * simSpeed) / stepsPerFrame;
    for (let i = 0; i < stepsPerFrame; i++) {
        updatePhysics(dt);
    }
}

// Diagnostics Plotting
let cvsOscilloscope, ctxOscilloscope;
let cvsMassSpec, ctxMassSpec;
let cvsIonAxes, ctxIonAxes;

function initDiagnosticCanvases() {
    cvsOscilloscope = document.getElementById("canvas-oscilloscope");
    ctxOscilloscope = cvsOscilloscope.getContext("2d");
    
    cvsMassSpec = document.getElementById("canvas-mass-spec");
    ctxMassSpec = cvsMassSpec.getContext("2d");

    cvsIonAxes = document.getElementById("canvas-ion-axes");
    ctxIonAxes = cvsIonAxes.getContext("2d");

    resizeCanvasElements();
    window.addEventListener('resize', resizeCanvasElements);
}

function resizeCanvasElements() {
    const dpr = window.devicePixelRatio || 1;
    
    const scaleCvs = (cvs) => {
        const rect = cvs.parentElement.getBoundingClientRect();
        cvs.width = rect.width * dpr;
        cvs.height = rect.height * dpr;
        cvs.style.width = `${rect.width}px`;
        cvs.style.height = `${rect.height}px`;
        const ctx = cvs.getContext("2d");
        ctx.resetTransform();
        ctx.scale(dpr, dpr);
    };

    scaleCvs(cvsOscilloscope);
    scaleCvs(cvsMassSpec);
    scaleCvs(cvsIonAxes);
}

function drawPlotGrid(ctx, w, h) {
    ctx.strokeStyle = "rgba(100, 116, 139, 0.16)";
    ctx.lineWidth = 1;
    
    for (let x = 40; x < w; x += 50) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
    }
    for (let y = 30; y < h; y += 40) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
    }
}

function updatePlots() {
    // Oscilloscope (Transient Signal)
    const wOsc = cvsOscilloscope.width / (window.devicePixelRatio || 1);
    const hOsc = cvsOscilloscope.height / (window.devicePixelRatio || 1);
    
    ctxOscilloscope.fillStyle = "#ffffff";
    ctxOscilloscope.fillRect(0, 0, wOsc, hOsc);
    drawPlotGrid(ctxOscilloscope, wOsc, hOsc);

    ctxOscilloscope.lineWidth = 1.5;
    ctxOscilloscope.strokeStyle = "#1769aa";
    ctxOscilloscope.shadowBlur = 0;

    ctxOscilloscope.beginPath();
    
    const validSignalCount = Math.min(signalCount, transientSize);

    const paddingX = 40;
    const paddingY = 20;
    const plotW = wOsc - paddingX - 10;
    const plotH = hOsc - paddingY - 10;

    // Auto-scale y-axis
    const displayPointCount = Math.min(validSignalCount, Math.max(2, Math.ceil(plotW * 2)));
    for (let point = 0; point < displayPointCount; point++) {
        const sampleIndex = Math.round((point / (displayPointCount - 1)) * Math.max(0, validSignalCount - 1));
        let x = paddingX + (point / (displayPointCount - 1)) * plotW;
        let valScaled = rawSignal[sampleIndex] * (plotH / (2.2 * signalAbsMax));
        let y = (paddingY + plotH / 2) - valScaled;

        if (y < paddingY) y = paddingY;
        if (y > paddingY + plotH) y = paddingY + plotH;

        if (point === 0) {
            ctxOscilloscope.moveTo(x, y);
        } else {
            ctxOscilloscope.lineTo(x, y);
        }
    }
    ctxOscilloscope.stroke();

    // Draw axes
    ctxOscilloscope.strokeStyle = "#52606d";
    ctxOscilloscope.lineWidth = 2;
    ctxOscilloscope.beginPath();
    ctxOscilloscope.moveTo(paddingX, paddingY);
    ctxOscilloscope.lineTo(paddingX, paddingY + plotH);
    ctxOscilloscope.lineTo(paddingX + plotW, paddingY + plotH);
    ctxOscilloscope.stroke();

    // Mass Spectrum (FFT)
    const wSpec = cvsMassSpec.width / (window.devicePixelRatio || 1);
    const hSpec = cvsMassSpec.height / (window.devicePixelRatio || 1);
    
    ctxMassSpec.fillStyle = "#ffffff";
    ctxMassSpec.fillRect(0, 0, wSpec, hSpec);
    drawPlotGrid(ctxMassSpec, wSpec, hSpec);

    // Transform the complete acquired transient. The transform grows with the
    // data and is cached so a long acquisition does not stall every redraw.
    refreshSpectrumFFT(validSignalCount);
    const fftMag = cachedFftMag.length > 0 ? cachedFftMag : new Float32Array(2);
    const fftSize = cachedFftSize || 4;

    const padX = 40;
    const padY = 20;
    const specW = wSpec - padX - 15;
    const specH = hSpec - padY - 20;

    ctxMassSpec.strokeStyle = "#52606d";
    ctxMassSpec.lineWidth = 2;
    ctxMassSpec.beginPath();
    ctxMassSpec.moveTo(padX, padY);
    ctxMassSpec.lineTo(padX, padY + specH);
    ctxMassSpec.lineTo(padX + specW, padY + specH);
    ctxMassSpec.stroke();

    // Calibrate frequency to m/z
    const df = 1.0 / (fftSize * detectorSampleInterval);
    const minMass = fullSpectrumRange.min;
    const maxMass = fullSpectrumRange.max;
    const viewMinMass = spectrumView.min;
    const viewMaxMass = spectrumView.max;

    // Estimate noise floor
    const minVisibleFrequency = (1.0 / (2.0 * Math.PI)) * Math.sqrt(V_final / maxMass);
    const maxVisibleFrequency = (1.0 / (2.0 * Math.PI)) * Math.sqrt(V_final / minMass);
    const firstVisibleBin = Math.max(1, Math.floor(minVisibleFrequency / df));
    const lastVisibleBin = Math.min(fftMag.length - 1, Math.ceil(maxVisibleFrequency / df));
    const noiseSamples = Array.from(fftMag.slice(firstVisibleBin, lastVisibleBin + 1));
    noiseSamples.sort((a, b) => a - b);
    const noiseFloor = noiseSamples.length > 0
        ? Math.max(noiseSamples[Math.floor(noiseSamples.length / 2)], 1e-12)
        : 1e-12;
    let maxVisibleMag = 1e-12;
    for (let i = firstVisibleBin; i <= lastVisibleBin; i++) {
        maxVisibleMag = Math.max(maxVisibleMag, fftMag[i]);
    }

    ctxMassSpec.strokeStyle = "#9a6700";
    ctxMassSpec.lineWidth = 1.8;
    ctxMassSpec.shadowBlur = 0;

    // Draw the Fourier response itself rather than replacing it with centroid
    // sticks. Sampling the zero-filled FFT at each screen column preserves the
    // raw peak envelope, shoulders, sidelobes, and noise at every zoom level.
    const spectrumY = (magnitude) => {
        const relativeDb = 20 * Math.log10(Math.max(magnitude, 1e-12) / maxVisibleMag);
        const normalizedHeight = Math.max(0, 1 + relativeDb / 60);
        return padY + specH - normalizedHeight * (specH - 10);
    };
    ctxMassSpec.beginPath();
    for (let pixel = 0; pixel <= Math.ceil(specW); pixel++) {
        const ratio = pixel / specW;
        const mass = viewMinMass + ratio * (viewMaxMass - viewMinMass);
        const frequency = (1.0 / (2.0 * Math.PI)) * Math.sqrt(V_final / mass);
        const binPosition = frequency / df;
        const lowerBin = Math.max(0, Math.min(fftMag.length - 1, Math.floor(binPosition)));
        const upperBin = Math.min(fftMag.length - 1, lowerBin + 1);
        const fraction = binPosition - lowerBin;
        const magnitude = fftMag[lowerBin] * (1 - fraction) + fftMag[upperBin] * fraction;
        const x = padX + pixel;
        const y = spectrumY(magnitude);
        if (pixel === 0) ctxMassSpec.moveTo(x, y);
        else ctxMassSpec.lineTo(x, y);
    }
    ctxMassSpec.stroke();

    // Peak centroiding via parabolic fit
    let detectedPeaks = [];
    const detectionThreshold = Math.max(noiseFloor * 6, maxVisibleMag * 0.03);
    if (validSignalCount >= 128) {
        for (let i = firstVisibleBin + 1; i < lastVisibleBin; i++) {
            if (fftMag[i] < detectionThreshold ||
                fftMag[i] <= fftMag[i - 1] || fftMag[i] <= fftMag[i + 1]) continue;

            const y0 = Math.log(Math.max(fftMag[i - 1], 1e-12));
            const y1 = Math.log(Math.max(fftMag[i], 1e-12));
            const y2 = Math.log(Math.max(fftMag[i + 1], 1e-12));
            const curvature = y0 - 2 * y1 + y2;
            const offset = curvature !== 0
                ? Math.max(-0.5, Math.min(0.5, 0.5 * (y0 - y2) / curvature))
                : 0;
            const frequency = (i + offset) * df;
            const mass = V_final / (4 * Math.PI * Math.PI * frequency * frequency);
            const magnitude = Math.exp(y1 - 0.25 * (y0 - y2) * offset);

            if (mass >= minMass && mass <= maxMass) {
                detectedPeaks.push({ mass, magnitude, snr: magnitude / noiseFloor });
            }
        }
    }

    detectedPeaks.sort((a, b) => b.magnitude - a.magnitude);
    detectedPeaks = detectedPeaks.slice(0, 12);
    spectrumPeakHitTargets = [];
    const selectedDetectedPeak = selectedSpectrumPeak === null
        ? null
        : detectedPeaks.reduce((closest, peak) => (
            closest === null || Math.abs(peak.mass - selectedSpectrumPeak) < Math.abs(closest.mass - selectedSpectrumPeak)
                ? peak
                : closest
        ), null);

    // Centroids remain as small hit targets for zoom selection, but the raw
    // trace above is the spectrum visualization.
    for (const peak of detectedPeaks) {
        if (peak.mass < viewMinMass || peak.mass > viewMaxMass) continue;

        const x = padX + ((peak.mass - viewMinMass) / (viewMaxMass - viewMinMass)) * specW;
        const y = spectrumY(peak.magnitude);
        const isSelected = peak === selectedDetectedPeak;
        ctxMassSpec.fillStyle = isSelected ? "#1769aa" : "#9a6700";
        ctxMassSpec.beginPath();
        ctxMassSpec.arc(x, y, isSelected ? 4 : 3, 0, 2 * Math.PI);
        ctxMassSpec.fill();
        if (isSelected) {
            ctxMassSpec.strokeStyle = "rgba(23, 105, 170, 0.45)";
            ctxMassSpec.lineWidth = 1;
            ctxMassSpec.beginPath();
            ctxMassSpec.moveTo(x, y + 5);
            ctxMassSpec.lineTo(x, padY + specH);
            ctxMassSpec.stroke();
        }
        peak.x = x;
        peak.y = y;
        spectrumPeakHitTargets.push({
            mass: peak.mass,
            x,
            top: y,
            bottom: padY + specH
        });
    }

    // Label peaks
    ctxMassSpec.fillStyle = "#765000";
    ctxMassSpec.font = "9px monospace";
    ctxMassSpec.textAlign = "center";
    for (const peak of detectedPeaks.filter(peak => peak.x !== undefined).slice(0, 6)) {
        ctxMassSpec.fillText(peak.mass.toFixed(1), peak.x, Math.max(padY + 9, peak.y - 5));
    }

    if (validSignalCount < 128) {
        ctxMassSpec.fillStyle = "#64748b";
        ctxMassSpec.font = "10px monospace";
        ctxMassSpec.textAlign = "center";
        const message = detectorAcquiring
            ? `Acquiring transient… ${validSignalCount} / ${transientSize}`
            : "Waiting for ion packet to settle…";
        ctxMassSpec.fillText(message, padX + specW / 2, padY + specH / 2);
    }

    const acquiredDuration = validSignalCount * detectorSampleInterval;
    const acquisitionStatus = document.getElementById("spectrum-acquisition-status");
    if (acquisitionStatus) {
        const resolutionAt300 = ((1 / (2 * Math.PI)) * Math.sqrt(V_final / 300) * acquiredDuration) / 2;
        const state = validSignalCount >= transientSize
            ? "complete"
            : detectorAcquiring ? "acquiring" : "waiting";
        acquisitionStatus.innerText = `Transient: ${(acquiredDuration / 1000).toFixed(3)} ms / ${(transientSize * detectorSampleInterval / 1000).toFixed(3)} ms (${state}) • R₃₀₀ ≈ ${Math.round(resolutionAt300)}`;
    }

    // Draw mass axis ticks
    ctxMassSpec.fillStyle = "var(--text-dim)";
    ctxMassSpec.font = "9px monospace";
    ctxMassSpec.textAlign = "center";
    const viewSpan = viewMaxMass - viewMinMass;
    const desiredTickStep = viewSpan / 5;
    const tickPower = Math.pow(10, Math.floor(Math.log10(desiredTickStep)));
    const tickScale = desiredTickStep / tickPower;
    const tickStep = (tickScale <= 1 ? 1 : tickScale <= 2 ? 2 : tickScale <= 5 ? 5 : 10) * tickPower;
    const firstTick = Math.ceil(viewMinMass / tickStep) * tickStep;
    const tickDecimals = tickStep < 1 ? Math.ceil(-Math.log10(tickStep)) : 0;
    for (let val = firstTick; val <= viewMaxMass + tickStep * 1e-6; val += tickStep) {
        let ratio = (val - viewMinMass) / viewSpan;
        let x = padX + ratio * specW;
        ctxMassSpec.fillText(val.toFixed(tickDecimals), x, padY + specH + 13);
        ctxMassSpec.beginPath();
        ctxMassSpec.moveTo(x, padY + specH);
        ctxMassSpec.lineTo(x, padY + specH + 4);
        ctxMassSpec.stroke();
    }

    // Ion Coordinates vs Time
    const wIon = cvsIonAxes.width / (window.devicePixelRatio || 1);
    const hIon = cvsIonAxes.height / (window.devicePixelRatio || 1);
    
    ctxIonAxes.fillStyle = "#ffffff";
    ctxIonAxes.fillRect(0, 0, wIon, hIon);
    drawPlotGrid(ctxIonAxes, wIon, hIon);

    let activeIon = ions[inspectedIonIndex];
    
    const pX = 40;
    const pY = 20;
    const iW = wIon - pX - 15;
    const iH = hIon - pY - 15;

    ctxIonAxes.strokeStyle = "#52606d";
    ctxIonAxes.lineWidth = 2;
    ctxIonAxes.beginPath();
    ctxIonAxes.moveTo(pX, pY);
    ctxIonAxes.lineTo(pX, pY + iH);
    ctxIonAxes.lineTo(pX + iW, pY + iH);
    ctxIonAxes.stroke();

    if (activeIon && activeIon.tSeriesX.length > 1) {
        const len = activeIon.tSeriesX.length;
        
        const drawCoordinate = (series, color) => {
            ctxIonAxes.strokeStyle = color;
            ctxIonAxes.lineWidth = 1.8;
            ctxIonAxes.beginPath();
            
            for (let i = 0; i < len; i++) {
                let x = pX + (i / 500) * iW;
                let valScaled = series[i] * (iH / 110);
                let y = (pY + iH / 2) - valScaled;

                if (y < pY) y = pY;
                if (y > pY + iH) y = pY + iH;

                if (i === 0) {
                    ctxIonAxes.moveTo(x, y);
                } else {
                    ctxIonAxes.lineTo(x, y);
                }
            }
            ctxIonAxes.stroke();
        };

        drawCoordinate(activeIon.tSeriesX, "#287a45");
        drawCoordinate(activeIon.tSeriesY, "#a23b72");
        drawCoordinate(activeIon.tSeriesZ, "#1769aa");
        
        document.getElementById("inspect-mz").innerText = activeIon.mz.toFixed(1);
        let statusStr = "Orbiting";
        let statusColor = "var(--accent-green)";
        if (!activeIon.active) {
            statusStr = `Collided (${activeIon.collisionStatus})`;
            statusColor = "var(--accent-red)";
        }
        const stNode = document.getElementById("inspect-status");
        stNode.innerText = statusStr;
        stNode.style.color = statusColor;
        
        document.getElementById("inspect-pos").innerText = `${activeIon.x.toFixed(1)}, ${activeIon.y.toFixed(1)}, ${activeIon.z.toFixed(1)}`;
        document.getElementById("inspect-vel").innerText = `${activeIon.vx.toFixed(2)}, ${activeIon.vy.toFixed(2)}, ${activeIon.vz.toFixed(2)}`;
    } else {
        ctxIonAxes.fillStyle = "rgba(148, 163, 184, 0.4)";
        ctxIonAxes.font = "11px 'Inter', sans-serif";
        ctxIonAxes.textAlign = "center";
        ctxIonAxes.fillText("No active ion selected or no data available.", pX + iW/2, pY + iH/2);
    }
}

// Ion management
function injectIons() {
    ions = [];
    nextIonId = 1;
    timeElapsed = 0.0;
    physicsStepCount = 0;
    
    rawSignal.fill(0);
    signalPtr = 0;
    signalCount = 0;
    signalAbsMax = 0.01;
    invalidateSpectrumCache();
    detectorSampleAccumulator = 0;
    detectorAcquiring = false;

    addIon(300, "#00f2ff", 30.0, 15.0);
    addIon(400, "#ffd700", 30.0, 15.0);
    addIon(500, "#ff007f", 30.0, 15.0);

    repopulateInspectSelect();
}

function reinjectCurrentIons() {
    timeElapsed = 0.0;
    physicsStepCount = 0;
    currentVoltage = V_init;
    
    rawSignal.fill(0);
    signalPtr = 0;
    signalCount = 0;
    signalAbsMax = 0.01;
    invalidateSpectrumCache();
    detectorSampleAccumulator = 0;
    detectorAcquiring = false;

    for (let ion of ions) {
        ion.x = ion.rInject;
        ion.y = 0.0;
        ion.z = ion.zInject;
        
        // Calculate tangential injection velocity for stable orbit
        ion.vy = 0.98 * Math.sqrt( (ion.q * currentVoltage) / (2 * ion.m) * (ion.x * ion.x - R_m * R_m) );
        ion.vx = 0.0;
        ion.vz = 0.0;
        
        ion.active = true;
        ion.collisionStatus = null;
        ion.hasLeftInjector = false;
        ion.history = [];
        ion.tSeriesX = [];
        ion.tSeriesY = [];
        ion.tSeriesZ = [];
    }
}

function addIon(mz, colorHex, rInject, zInject) {
    let qVal = 1.0;
    let mVal = mz * qVal;
    
    let x0 = rInject;
    let y0 = 0.0;
    let z0 = zInject;

    // Tangential injection velocity for stable orbit
    let vy0 = 0.98 * Math.sqrt( (qVal * currentVoltage) / (2 * mVal) * (x0 * x0 - R_m * R_m) );
    let vx0 = 0.0;
    let vz0 = 0.0;

    ions.push({
        id: nextIonId++,
        mz: mz,
        q: qVal,
        m: mVal,
        x: x0,
        y: y0,
        z: z0,
        vx: vx0,
        vy: vy0,
        vz: vz0,
        rInject: rInject,
        zInject: zInject,
        hasLeftInjector: false,
        color: colorHex,
        active: true,
        collisionStatus: null,
        history: [],
        tSeriesX: [],
        tSeriesY: [],
        tSeriesZ: []
    });
}

function repopulateInspectSelect() {
    let select = document.getElementById("select-inspect-ion");
    select.innerHTML = "";
    ions.forEach((ion, index) => {
        let opt = document.createElement("option");
        opt.value = index;
        let colorName = ion.color === "#00f2ff" ? "Cyan" : (ion.color === "#ffd700" ? "Yellow" : "Pink");
        if (ion.id > 3) colorName = "Custom";
        opt.innerText = `Ion ${ion.id} (m/z ${Math.round(ion.mz)} - ${colorName})`;
        select.appendChild(opt);
    });
    
    inspectedIonIndex = Math.min(inspectedIonIndex, ions.length - 1);
    if (inspectedIonIndex < 0) inspectedIonIndex = 0;
    select.value = inspectedIonIndex;
}

// 3D Rendering (p5.js)
const p5Sketch = (p) => {
    let font;

    p.setup = () => {
        const holder = document.getElementById("p5-canvas-holder");
        const canvas = p.createCanvas(holder.clientWidth, holder.clientHeight, p.WEBGL);
        canvas.parent(holder);
        p.pixelDensity(1);
        
        p.debugMode;
        
        window.addEventListener('resize', () => {
            const h = document.getElementById("p5-canvas-holder");
            p.resizeCanvas(h.clientWidth, h.clientHeight);
        });
    };

    p.draw = () => {
        // A cool, neutral backdrop gives the metal enough contrast without
        // competing with the coloured ion trails.
        p.background(236, 240, 244);

        if (isPlaying) {
            advanceSimulationFrame();
        }

        // Studio-style lighting: a warm key defines the electrode curvature,
        // while the cool fill and rim light keep the cutaway legible.
        p.ambientLight(58, 63, 70);
        p.directionalLight(255, 244, 220, -0.45, 0.35, -1.0);
        p.directionalLight(105, 145, 184, 0.7, -0.25, 0.45);
        p.pointLight(255, 213, 155, -190, -130, 240);
        p.orbitControl(1, 1, 0.1);
        
        p.rotateX(-0.5);
        p.rotateY(0.7);

        if (showGrid) {
            drawCoordinateSystemAxes(p);
        }

        const drawScale = 4.0;

        // Central spindle: warm plated metal rather than a flat yellow fill.
        p.push();
        p.noStroke();
        let innerAlpha = p.map(innerOpacity, 0, 100, 0, 255);
        p.specularMaterial(190, 137, 42, innerAlpha);
        p.shininess(88);
        drawRevolvedSurface3D(p, fullInnerProfile, 0, p.TWO_PI, 36, drawScale);
        p.pop();

        // Outer electrodes (split barrel)
        let outerAlpha = p.map(outerOpacity, 0, 100, 0, 255);
        if (outerAlpha > 0) {
            p.push();
            p.noStroke();
            p.specularMaterial(112, 139, 158, outerAlpha);
            p.shininess(115);

            let startTheta = p.PI - cutawayAngle / 2;
            let endTheta = p.PI + cutawayAngle / 2;

            let leftBarrelProfile = fullOuterProfile.filter(pt => pt.z < -0.5);
            let rightBarrelProfile = fullOuterProfile.filter(pt => pt.z > 0.5);

            drawRevolvedSurface3D(p, leftBarrelProfile, startTheta, endTheta, 32, drawScale);
            drawRevolvedSurface3D(p, rightBarrelProfile, startTheta, endTheta, 32, drawScale);
            p.pop();
        }

        // Injector nozzle
        p.push();
        p.translate(15.0 * drawScale, 30.0 * drawScale, -7.5 * drawScale);
        p.rotateX(p.HALF_PI);
        p.specularMaterial(94, 103, 112);
        p.shininess(125);
        p.noStroke();
        p.cylinder(2.5 * drawScale, 15.0 * drawScale);
        p.pop();

        // Ions and trails
        for (let ion of ions) {
            if (!ion.active) continue;

            if (ion.history.length > 1) {
                p.push();
                p.noFill();
                for (let j = 0; j < ion.history.length - 1; j++) {
                    let pt1 = ion.history[j];
                    let pt2 = ion.history[j+1];
                    
                    let alpha = p.map(j, 0, ion.history.length - 1, 0, 255);
                    let c = p.color(ion.color);
                    c.setAlpha(alpha);
                    
                    p.stroke(c);
                    p.strokeWeight(p.map(j, 0, ion.history.length - 1, 0.4, 2.5));
                    p.line(
                        pt1.z * drawScale, pt1.x * drawScale, pt1.y * drawScale,
                        pt2.z * drawScale, pt2.x * drawScale, pt2.y * drawScale
                    );
                }
                p.pop();
            }

            p.push();
            p.translate(ion.z * drawScale, ion.x * drawScale, ion.y * drawScale);
            p.noStroke();
            p.ambientMaterial(p.color(ion.color));
            p.fill(p.color(ion.color));
            p.sphere(3.5);
            p.pop();
        }
        
        // Collision points
        for (let ion of ions) {
            if (!ion.active && ion.history.length > 0) {
                let lastPt = ion.history[ion.history.length - 1];
                p.push();
                p.translate(lastPt.z * drawScale, lastPt.x * drawScale, lastPt.y * drawScale);
                p.noStroke();
                p.fill(255, 49, 49, 120);
                p.sphere(5);
                p.pop();
            }
        }
    };
};

// Axes drawing helper
function drawCoordinateSystemAxes(p) {
    p.push();
    p.strokeWeight(3);
    
    p.stroke(23, 105, 170, 180); // Z-axis (Blue)
    p.line(-220, 0, 0, 220, 0, 0);

    p.stroke(40, 122, 69, 180); // X-axis (Green)
    p.line(0, -180, 0, 0, 180, 0);

    p.stroke(162, 59, 114, 180); // Y-axis (Magenta)
    p.line(0, 0, -180, 0, 0, 180);
    p.pop();
}

// Revolved 3D surface generator
function drawRevolvedSurface3D(p, profile, startTheta, endTheta, stepsTheta, drawScale) {
    const applyProfileNormal = (index, cosTheta, sinTheta) => {
        const previous = profile[Math.max(0, index - 1)];
        const next = profile[Math.min(profile.length - 1, index + 1)];
        const axialDelta = next.z - previous.z;
        const radialDelta = next.r - previous.r;

        // Surface-of-revolution normal for x=z, y=r*cos(theta), z=r*sin(theta).
        // Supplying it explicitly gives WebGL a smooth highlight across the
        // machined profile instead of lighting every strip as a flat facet.
        let nx = -radialDelta;
        let ny = axialDelta * cosTheta;
        let nz = axialDelta * sinTheta;
        const length = Math.hypot(nx, ny, nz) || 1;
        p.normal(nx / length, ny / length, nz / length);
    };

    for (let i = 0; i < profile.length - 1; i++) {
        let p1 = profile[i];
        let p2 = profile[i+1];
        
        p.beginShape(p.TRIANGLE_STRIP);
        for (let j = 0; j <= stepsTheta; j++) {
            let theta = p.map(j, 0, stepsTheta, startTheta, endTheta);
            let cosT = p.cos(theta);
            let sinT = p.sin(theta);
            
            let x1 = p1.z * drawScale;
            let y1 = p1.r * cosT * drawScale;
            let z1 = p1.r * sinT * drawScale;
            applyProfileNormal(i, cosT, sinT);
            p.vertex(x1, y1, z1);
            
            let x2 = p2.z * drawScale;
            let y2 = p2.r * cosT * drawScale;
            let z2 = p2.r * sinT * drawScale;
            applyProfileNormal(i + 1, cosT, sinT);
            p.vertex(x2, y2, z2);
        }
        p.endShape();
    }
}

// UI Events & setup
function setupUI() {
    // Tab handling
    const tabs = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");

    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            tabs.forEach(t => t.classList.remove("active"));
            tabContents.forEach(c => c.classList.remove("active"));

            tab.classList.add("active");
            const targetTab = tab.getAttribute("data-tab");
            document.getElementById(targetTab).classList.add("active");
            
            resizeCanvasElements();
            updatePlots();
        });
    });

    // Playback controls
    const btnPlay = document.getElementById("btn-play");
    btnPlay.addEventListener("click", () => {
        isPlaying = !isPlaying;
        btnPlay.innerText = isPlaying ? "Pause" : "Play";
        btnPlay.classList.toggle("btn-primary");
    });

    document.getElementById("btn-step").addEventListener("click", () => {
        if (!isPlaying) {
            advanceSimulationFrame();
            updatePlots();
        }
    });

    const spectrumZoomInButton = document.getElementById("btn-spectrum-zoom-in");
    const spectrumResetButton = document.getElementById("btn-spectrum-reset");
    const resetSpectrumZoom = () => {
        spectrumView = { ...fullSpectrumRange };
        selectedSpectrumPeak = null;
        spectrumZoomInButton.hidden = true;
        spectrumZoomInButton.disabled = false;
        spectrumResetButton.hidden = true;
        updatePlots();
    };

    const setSpectrumViewAroundPeak = (mass, span) => {
        const fullSpan = fullSpectrumRange.max - fullSpectrumRange.min;
        const boundedSpan = Math.min(span, fullSpan);
        let min = mass - boundedSpan / 2;
        let max = mass + boundedSpan / 2;
        if (min < fullSpectrumRange.min) {
            max += fullSpectrumRange.min - min;
            min = fullSpectrumRange.min;
        }
        if (max > fullSpectrumRange.max) {
            min -= max - fullSpectrumRange.max;
            max = fullSpectrumRange.max;
        }
        spectrumView = { min, max };
    };

    const massSpecCanvas = document.getElementById("canvas-mass-spec");
    massSpecCanvas.addEventListener("click", (event) => {
        if (spectrumPeakHitTargets.length === 0) return;

        const rect = massSpecCanvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        const selected = spectrumPeakHitTargets
            .map(peak => ({
                ...peak,
                distance: Math.abs(peak.x - x)
            }))
            .filter(peak => peak.distance <= 12 && y >= peak.top - 10 && y <= peak.bottom + 10)
            .sort((a, b) => a.distance - b.distance)[0];

        if (!selected) return;

        selectedSpectrumPeak = selected.mass;
        setSpectrumViewAroundPeak(selectedSpectrumPeak, 50);
        spectrumZoomInButton.hidden = false;
        spectrumZoomInButton.disabled = false;
        spectrumResetButton.hidden = false;
        updatePlots();
    });
    spectrumZoomInButton.addEventListener("click", () => {
        if (selectedSpectrumPeak === null) return;

        const minimumSpan = 0.01;
        const currentSpan = spectrumView.max - spectrumView.min;
        const nextSpan = Math.max(minimumSpan, currentSpan / 2);
        setSpectrumViewAroundPeak(selectedSpectrumPeak, nextSpan);
        spectrumZoomInButton.disabled = nextSpan <= minimumSpan;
        updatePlots();
    });
    spectrumResetButton.addEventListener("click", resetSpectrumZoom);

    const processingControls = [
        ["spectrum-remove-baseline", value => { spectrumRemoveBaseline = value; }],
        ["spectrum-hann-window", value => { spectrumHannWindow = value; }],
        ["spectrum-zero-fill", value => { spectrumZeroFill = value; }]
    ];
    for (const [id, applyValue] of processingControls) {
        document.getElementById(id).addEventListener("change", event => {
            applyValue(event.target.checked);
            invalidateSpectrumCache();
            updatePlots();
        });
    }

    document.getElementById("btn-reset").addEventListener("click", () => {
        injectIons();
        updatePlots();
    });

    document.getElementById("btn-inject").addEventListener("click", () => {
        reinjectCurrentIons();
        updatePlots();
    });

    // Controls & sliders
    const speedInput = document.getElementById("input-sim-speed");
    const applySimulationSpeed = () => {
        const nextSpeed = Number(speedInput.value);
        if (Number.isFinite(nextSpeed) && nextSpeed > 0) {
            simSpeed = nextSpeed;
            speedInput.setCustomValidity("");
        } else {
            speedInput.setCustomValidity("Enter a simulation speed greater than 0.");
        }
    };
    speedInput.addEventListener("input", applySimulationSpeed);
    speedInput.addEventListener("change", () => {
        applySimulationSpeed();
        if (!speedInput.checkValidity()) {
            speedInput.reportValidity();
            speedInput.value = String(simSpeed);
            speedInput.setCustomValidity("");
        }
    });

    const sliderNoise = document.getElementById("slider-noise");
    sliderNoise.addEventListener("input", (e) => {
        detectorNoise = parseFloat(e.target.value);
        document.getElementById("lbl-noise").innerText = `${detectorNoise}%`;
    });

    const toggleRamping = document.getElementById("toggle-voltage-ramping");
    toggleRamping.addEventListener("change", (e) => {
        voltageRamping = e.target.checked;
    });

    const sliderVInit = document.getElementById("slider-v-init");
    sliderVInit.addEventListener("input", (e) => {
        V_init = parseFloat(e.target.value);
        document.getElementById("lbl-v-init").innerText = `${V_init} V`;
    });

    const sliderVFinal = document.getElementById("slider-v-final");
    sliderVFinal.addEventListener("input", (e) => {
        V_final = parseFloat(e.target.value);
        document.getElementById("lbl-v-final").innerText = `${V_final} V`;
    });

    const sliderTRamp = document.getElementById("slider-t-ramp");
    sliderTRamp.addEventListener("input", (e) => {
        t_ramp_time = parseFloat(e.target.value);
        document.getElementById("lbl-t-ramp").innerText = `${t_ramp_time.toFixed(1)} \u03BCs`;
    });

    const sliderCutaway = document.getElementById("slider-cutaway");
    sliderCutaway.addEventListener("input", (e) => {
        let degrees = parseInt(e.target.value);
        cutawayAngle = (degrees / 180.0) * Math.PI;
        document.getElementById("lbl-cutaway").innerText = `${degrees}°`;
    });

    const sliderOpacityOuter = document.getElementById("slider-opacity-outer");
    sliderOpacityOuter.addEventListener("input", (e) => {
        outerOpacity = parseInt(e.target.value);
        document.getElementById("lbl-opacity-outer").innerText = `${outerOpacity}%`;
    });

    const sliderOpacityInner = document.getElementById("slider-opacity-inner");
    sliderOpacityInner.addEventListener("input", (e) => {
        innerOpacity = parseInt(e.target.value);
        document.getElementById("lbl-opacity-inner").innerText = `${innerOpacity}%`;
    });

    document.getElementById("btn-toggle-grid").addEventListener("click", (e) => {
        showGrid = !showGrid;
        e.target.classList.toggle("btn-primary");
    });

    const selectIon = document.getElementById("select-inspect-ion");
    selectIon.addEventListener("change", (e) => {
        inspectedIonIndex = parseInt(e.target.value);
        updatePlots();
    });

    document.getElementById("btn-add-ion").addEventListener("click", () => {
        const mzInput = document.getElementById("input-add-mz");
        const colorSelect = document.getElementById("select-add-color");
        const radiusInput = document.getElementById("input-add-radius");
        const zInput = document.getElementById("input-add-z");

        const mz = parseFloat(mzInput.value);
        const rInject = parseFloat(radiusInput.value);
        const zInject = parseFloat(zInput.value);
        
        let colorHex = "#ffd700";
        switch (colorSelect.value) {
            case "cyan": colorHex = "#00f2ff"; break;
            case "magenta": colorHex = "#ff007f"; break;
            case "green": colorHex = "#39ff14"; break;
            case "orange": colorHex = "#ff5500"; break;
        }

        addIon(mz, colorHex, rInject, zInject);
        repopulateInspectSelect();
        
        inspectedIonIndex = ions.length - 1;
        document.getElementById("select-inspect-ion").value = inspectedIonIndex;
    });

    document.getElementById("btn-camera-reset").addEventListener("click", () => {
        if (p5Instance) {
            p5Instance.resetMatrix();
        }
    });
}

// Main Initialization
let p5Instance;

function initApp() {
    generateElectrodeProfiles();
    injectIons();
    setupUI();
    initDiagnosticCanvases();

    p5Instance = new p5(p5Sketch);

    // Diagnostic plots update loop (~24 fps)
    setInterval(() => {
        if (isPlaying) {
            updatePlots();
        }
    }, 40);
}

window.addEventListener('DOMContentLoaded', initApp);
