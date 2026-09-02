// `arch` is an alternative spelling for triple + chip. It is split apart
// because the attribute feeds the TargetMachine, whose -mcpu only accepts a
// bare processor name, and the triple is normalized to the new-style spelling
// that names the GPU's subarch.

// Every spelling of gfx90a lands on the same attribute, whichever triple it
// arrived with.
// RUN: mlir-opt %s --rocdl-attach-target='arch=gfx90a' \
// RUN: | FileCheck %s
// RUN: mlir-opt %s --rocdl-attach-target='arch=amdgcn-amd-amdhsa--gfx90a' \
// RUN: | FileCheck %s
// RUN: mlir-opt %s --rocdl-attach-target='arch=amdgpu9.0a-amd-amdhsa--gfx90a' \
// RUN: | FileCheck %s

// The legacy split options still attach what they always did, so migrating is
// opt-in: only `arch` normalizes the triple or consults the target for the
// wavefront size.
// RUN: mlir-opt %s \
// RUN:   --rocdl-attach-target='triple=amdgcn-amd-amdhsa chip=gfx90a' \
// RUN: | FileCheck %s --check-prefix=LEGACY

// An explicit `features` is passed through untouched.
// RUN: mlir-opt %s --rocdl-attach-target='arch=gfx90a features=+dpp' \
// RUN: | FileCheck %s --check-prefix=FEATURES

// The processor is the more specific fact, so a family triple normalizes down
// to the subarch of the GPU it resolved to.
// RUN: mlir-opt %s --rocdl-attach-target='arch=amdgpu9.4-amd-amdhsa--gfx950' \
// RUN: | FileCheck %s --check-prefix=FAMILY

// A generic target normalizes to its family subarch.
// RUN: mlir-opt %s --rocdl-attach-target='arch=gfx9-4-generic' \
// RUN: | FileCheck %s --check-prefix=GENERIC

// The wavefront size comes from the target, so a wave32-only chip gets
// no_wave64 rather than the `wave64` option's default.
// RUN: mlir-opt %s --rocdl-attach-target='arch=gfx1250' \
// RUN: | FileCheck %s --check-prefix=WAVE32

// A dual-mode chip defaults to wave32 and honours an explicit `wavesize`. The
// size lands in both the flags (for the device libraries) and the features (for
// the TargetMachine), which must agree.
// RUN: mlir-opt %s --rocdl-attach-target='arch=gfx1030' \
// RUN: | FileCheck %s --check-prefix=WAVE32-RDNA
// RUN: mlir-opt %s --rocdl-attach-target='arch=gfx1030 wavesize=64' \
// RUN: | FileCheck %s --check-prefix=WAVE64-RDNA

// Errors have no source location, so they are matched on stderr.
// RUN: not mlir-opt %s --rocdl-attach-target='arch=gfx999' 2>&1 \
// RUN: | FileCheck %s --check-prefix=ERR
// ERR: 'gfx999' is not a valid AMDGPU architecture

// A triple naming no GPU cannot be attached: the attribute requires a chip.
// RUN: not mlir-opt %s --rocdl-attach-target='arch=amdgcn-amd-amdhsa' 2>&1 \
// RUN: | FileCheck %s --check-prefix=NOGPU
// NOGPU: 'amdgcn-amd-amdhsa' names no GPU

// xnack/sramecc are module flags in the backend, not subtarget features, and
// #rocdl.target has nowhere to carry them: refuse rather than emit a feature
// string that AMDGPUAsmPrinter rejects at serialization.
// RUN: not mlir-opt %s --rocdl-attach-target='arch=gfx90a:xnack+' 2>&1 \
// RUN: | FileCheck %s --check-prefix=XNACK
// XNACK: the 'xnack' target-ID modifier cannot be attached
// XNACK-SAME: 'amdgpu.xnack' module flag

// RUN: not mlir-opt %s --rocdl-attach-target='arch=gfx90a:sramecc-' 2>&1 \
// RUN: | FileCheck %s --check-prefix=SRAMECC
// SRAMECC: the 'sramecc' target-ID modifier cannot be attached

module attributes {gpu.container_module} {

// CHECK-LABEL: @rocdl_module
// CHECK-SAME: [#rocdl.target<triple = "amdgpu9.0a-amd-amdhsa", chip = "gfx90a">]

// LEGACY-LABEL: @rocdl_module
// LEGACY-SAME: [#rocdl.target<chip = "gfx90a">]

// FEATURES-LABEL: @rocdl_module
// FEATURES-SAME: [#rocdl.target<triple = "amdgpu9.0a-amd-amdhsa", chip = "gfx90a", features = "+dpp">]

// FAMILY-LABEL: @rocdl_module
// FAMILY-SAME: [#rocdl.target<triple = "amdgpu9.50-amd-amdhsa", chip = "gfx950">]

// GENERIC-LABEL: @rocdl_module
// GENERIC-SAME: [#rocdl.target<triple = "amdgpu9.4-amd-amdhsa", chip = "gfx9-4-generic">]

// WAVE32-LABEL: @rocdl_module
// WAVE32-SAME: [#rocdl.target<triple = "amdgpu12.50-amd-amdhsa", chip = "gfx1250", flags = {no_wave64}>]

// WAVE32-RDNA-LABEL: @rocdl_module
// WAVE32-RDNA-SAME: [#rocdl.target<triple = "amdgpu10.30-amd-amdhsa", chip = "gfx1030", features = "+wavefrontsize32", flags = {no_wave64}>]

// WAVE64-RDNA-LABEL: @rocdl_module
// WAVE64-RDNA-SAME: [#rocdl.target<triple = "amdgpu10.30-amd-amdhsa", chip = "gfx1030", features = "+wavefrontsize64">]
gpu.module @rocdl_module {
}

}
