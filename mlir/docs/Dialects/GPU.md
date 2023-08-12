# 'gpu' Dialect

Note: this dialect is more likely to change than others in the near future; use
with caution.

This dialect provides middle-level abstractions for launching GPU kernels
following a programming model similar to that of CUDA or OpenCL. It provides
abstractions for kernel invocations (and may eventually provide those for device
management) that are not present at the lower level (e.g., as LLVM IR intrinsics
for GPUs). Its goal is to abstract away device- and driver-specific
manipulations to launch a GPU kernel and provide a simple path towards GPU
execution from MLIR. It may be targeted, for example, by DSLs using MLIR. The
dialect uses `gpu` as its canonical prefix.

[TOC]

## Memory attribution

Memory buffers are defined at the function level, either in "gpu.launch" or in
"gpu.func" ops. This encoding makes it clear where the memory belongs and makes
the lifetime of the memory visible. The memory is only accessible while the
kernel is launched/the function is currently invoked. The latter is more strict
than actual GPU implementations but using static memory at the function level is
just for convenience. It is also always possible to pass pointers to the
workgroup memory into other functions, provided they expect the correct memory
space.

The buffers are considered live throughout the execution of the GPU function
body. The absence of memory attribution syntax means that the function does not
require special buffers. Rationale: although the underlying models declare
memory buffers at the module level, we chose to do it at the function level to
provide some structuring for the lifetime of those buffers; this avoids the
incentive to use the buffers for communicating between different kernels or
launches of the same kernel, which should be done through function arguments
instead; we chose not to use `alloca`-style approach that would require more
complex lifetime analysis following the principles of MLIR that promote
structure and representing analysis results in the IR.

## GPU Compilation
### Deprecation notice
The `--gpu-to-(cubin|hsaco)` passes will be deprecated in a future release.

### Compilation overview
The compilation process in the GPU dialect has two main stages: GPU module
serialization and offloading operations translation. Together these stages can
produce GPU binaries and the necessary code to execute them.

An example of how the compilation workflow look is:

```
mlir-opt example.mlir                   \
  --pass-pipeline="builtin.module(      \
    nvvm-attach-target{chip=sm_90 O=3}, \ # Attach an NVVM target to a gpu.module op.
    gpu.module(convert-gpu-to-nvvm),    \ # Convert GPU to NVVM.
    gpu-to-llvm,                        \ # Convert GPU to LLVM.
    gpu-module-to-binary                \ # Serialize GPU modules to binaries.
  )" -o example-nvvm.mlir
mlir-translate example-nvvm.mlir        \
  --mlir-to-llvmir                      \ # Obtain the translated LLVM IR.
  -o example.ll
```

### Module serialization
Attributes implementing the GPU Target Attribute Interface handle the
serialization process and are called Target attributes. These attributes can be
attached to GPU Modules indicating the serialization scheme to compile the
module into a binary string.

The `gpu-module-to-binary` pass searches for all nested GPU modules and
serializes the module using the target attributes attached to the module,
producing a binary with an object for every target.

Example:
```
// Input:
gpu.module @kernels [#nvvm.target<chip = "sm_90">, #nvvm.target<chip = "sm_60">] {
  ...
}
// mlir-opt --gpu-module-to-binary:
gpu.binary @kernels [
  #gpu.object<#nvvm.target<chip = "sm_90">, "sm_90 cubin">,
  #gpu.object<#nvvm.target<chip = "sm_60">, "sm_60 cubin">
]
```

### Offloading LLVM translation
Attributes implementing the GPU Offloading LLVM Translation Attribute Interface
handle the translation of GPU binaries and kernel launches into LLVM
instructions and are called Offloading attributes. These attributes are
attached to GPU binary operations.

During the LLVM translation process, GPU binaries get translated using the
scheme provided by the Offloading attribute, translating the GPU binary into
LLVM instructions. Meanwhile, Kernel launches are translated by searching the
appropriate binary and invoking the procedure provided by the Offloading
attribute in the binary for translating kernel launches into LLVM instructions.

Example:
```
// Input:
// Binary with multiple objects but selecting the second one for embedding.
gpu.binary @binary <#gpu.select_object<#rocdl.target<chip = "gfx90a">>> [
    #gpu.object<#nvvm.target, "NVPTX">,
    #gpu.object<#rocdl.target<chip = "gfx90a">, "AMDGPU">
  ]
llvm.func @foo() {
  ...
  // Launching a kernel inside the binary.
  gpu.launch_func @binary::@func blocks in (%0, %0, %0)
                                 threads in (%0, %0, %0) : i64
                                 dynamic_shared_memory_size %2
                                 args(%1 : i32, %1 : i32)
  ...
}
// mlir-translate --mlir-to-llvmir:
@binary_bin_cst = internal constant [6 x i8] c"AMDGPU", align 8
@binary_func_kernel_name = private unnamed_addr constant [7 x i8] c"func\00", align 1
...
define void @foo() {
  ...
  %module = call ptr @mgpuModuleLoad(ptr @binary_bin_cst)
  %kernel = call ptr @mgpuModuleGetFunction(ptr %module, ptr @binary_func_kernel_name)
  call void @mgpuLaunchKernel(ptr %kernel, ...) ; Launch the kernel
  ...
  call void @mgpuModuleUnload(ptr %module)
  ...
}
...
```

### The binary operation
From a semantic point of view, GPU binaries allow the implementation of many
concepts, from simple object files to fat binaries. By default, the binary
operation uses the `#gpu.select_object` offloading attribute; this attribute
embeds a single object in the binary as a global string, see the attribute docs
for more information.

## Operations

[include "Dialects/GPUOps.md"]
