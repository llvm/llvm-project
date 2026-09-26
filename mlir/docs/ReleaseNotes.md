# MLIR Release Notes

This document tries to provide some context about MLIR important changes in the
context of LLVM releases. It is updated on a best effort basis.

At the moment the MLIR community does not qualify the LLVM release branch
specifically, it is a snapshot of the MLIR development at the time of the release.

[TOC]

## LLVM 24

### GPU/AMDGPU Changes

- `mlir::amdgpu::Chipset` is deprecated in favour of `mlir::ROCDL::TargetInfo`,
  which describes a target by its triple, subarch, and the resolved set of
  target features from LLVM's own tables. Lowerings should ask whether a target
  has a feature rather than inaccurately compare chipset versions.
  `TargetInfo` also represents generic targets such as `gfx9-4-generic` and, unlike
  `Chipset`, explicitly stores the wavesize for targets where it is configurable.
- The `chipset` option in AMDGPU passes is renamed to an `arch` option, which uses
  Clang target naming syntax. It accepts a GPU name with optional
  modifiers (`gfx942`, `gfx942:xnack+`, `gfx9-4-generic`), a triple
  (`amdgpu9.42-amd-amdhsa`), or a full target ID
  (`amdgpu9.42-amd-amdhsa--gfx90a:sramecc+:xnack-`, which is what `rocminfo` prints
  for a device's ISA). `chipset` or `chip` remain as compatibility names.
  The default arch is `invalid`, so a target must be passed
  explicitly, removing the old "fallback" `gfx000` GPU.
- Wavefront size is not a target-ID feature, so `convert-gpu-to-rocdl` takes it
  as a separate `wavesize` option (32, 64, or 0 for the architecture's
  default). The `wave64` flag on `gpu-lower-to-rocdl-pipeline` and on
  `rocdl-attach-target` is likewise replaced by the same `wavesize` option,
  which has the same allowed values.
- In keeping with broader LLVM changes, `xnack` and `sramecc` are no longer
  architecture features but module flags. In keeping with Clang, the target
  specifier still includes these xnack/sramecc flags where they're configurable,
  but lowering passes now convert these to module flags. Downstream users should
  call `migrateArchFeaturesToModuleFlags` to lower these attributes in custom
  pipelines.
- `rocdl-attach-target` gains `arch` alongside its existing `triple`, `chip` and
  `features`. When `arch` is given, it overrides `triple` and `chip`, and handles
  xnack/sramecc modifier migration.

## LLVM 21

### GPU/NVVM Changes

- The default NVVM target architecture has been changed from `sm_50` to `sm_75`.
  `sm_75` is the oldest GPU variant compatible with the widest range of recent
  major CUDA Toolkit versions (11/12/13). This affects the `NVVMTargetAttr`,
  `GpuNVVMAttachTarget` pass, and the `gpu-lower-to-nvvm-pipeline`.

## LLVM 20

All the MLIR runners other than `mlir-cpu-runner` have been removed, as their functionality has been merged into it, and it has been renamed to `mlir-runner`.

## LLVM 18

### Properties: beyond attributes

See LLVM 17 notes below. The Dialect option `let usePropertiesForAttributes = 1;` is
now the default. You can set it to 0 to revert to the previous behavior. This will be
removed in LLVM 19.

## LLVM 17

See also the [deprecations and refactoring](https://mlir.llvm.org/deprecation/) doc.

### Bytecode

MLIR now support a [bytecode serialization](https://mlir.llvm.org/docs/BytecodeFormat/)
with versionning compatibility allowing 2 ways compatibility scheme, and lazy-loading
capabilities.

### Properties: beyond attributes

This is a new mechanism to implement storage for operations without having to
use attributes. You can opt-in to use Properties for ODS inherent attributes
using `let usePropertiesForAttributes = 1;` in your dialect definition (the flag
will be default in the next release). See
[slides](https://mlir.llvm.org/OpenMeetings/2023-02-09-Properties.pdf) and
[recording](https://youtu.be/7ofnlCFzlqg) of the open meeting presentation for
details.

### Action: Tracing and Debugging MLIR-based Compilers

[Action](https://mlir.llvm.org/docs/ActionTracing/) is a new mechanism to
encapsulate any transformation of any granularity in a way that can be
intercepted by the framework for debugging or tracing purposes, including
skipping a transformation programmatically (think about “compiler fuel” or
“debug counters” in LLVM). As such, “executing a pass” is an Action, so is “try
to apply one canonicalization pattern”, or “tile this loop”.

[slides](https://mlir.llvm.org/OpenMeetings/2023-02-23-Actions.pdf) and
[recording](https://youtu.be/ayQSyekVa3c) of the open meeting presentation for
details.

### Transform Dialect

See this [EuroLLVM talk](https://www.youtube.com/watch?v=P4gUj3QtH_Y&t=1s) and
[the online tutorial](https://mlir.llvm.org/docs/Tutorials/transform/).

### Others

- There is now support for
  "[distinct attributes](https://mlir.llvm.org/docs/Dialects/Builtin/#distinctattribute)".
- "Resources" (a way to store data outside the MLIR context) and "configuration"
  can now be serialized alongside the IR.
