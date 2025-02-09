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

This dialect also abstracts away primitives commonly available in GPU code, such
as with `gpu.thread_id` (an operation that returns the ID of threads within
a thread block/workgroup along a given dimension). While the compilation
pipelines documented below expect such code to live inside a `gpu.module` and
`gpu.func`, these intrinsic wrappers may be used outside of this context.

Intrinsic-wrapping operations should not expect that they have a parent of type
`gpu.func`. However, operations that deal in compiling and launching GPU functions,
like `gpu.launch_func` or `gpu.binary` may assume that the dialect's full layering
is being used.

[TOC]

## GPU address spaces

The GPU dialect exposes the `gpu.address_space` attribute, which currently has
three values: `global`, `workgroup`, and `private`.

These address spaces represent the types of buffer commonly seen in GPU compilation.
`global` memory is memory that resides in the GPU's global memory. `workgroup`
memory is a limited, per-workgroup resource: all threads in a workgroup/thread
block access the same values in `workgroup` memory. Finally, `private` memory is
used to represent `alloca`-like buffers that are private to a single thread/workitem.

These address spaces may be used as the `memorySpace` attribute on `memref` values.
The `gpu.module`/`gpu.func` compilation pipeline will lower such memory space
usages to the correct address spaces on target platforms. Memory attributions should be
created with the correct memory space on the memref.

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
### Compilation overview
The compilation process in the GPU dialect has two main stages: GPU module
serialization and offloading operations translation. Together these stages can
produce GPU binaries and the necessary code to execute them.

An example of how the compilation workflow look is:

```
mlir-opt example.mlir                   \
  --pass-pipeline="builtin.module(      \
    gpu-kernel-outlining,               \ # Outline gpu.launch body to a kernel.
    nvvm-attach-target{chip=sm_90 O=3}, \ # Attach an NVVM target to a gpu.module op.
    gpu.module(convert-gpu-to-nvvm),    \ # Convert GPU to NVVM.
    gpu-to-llvm,                        \ # Convert GPU to LLVM.
    gpu-module-to-binary                \ # Serialize GPU modules to binaries.
  )" -o example-nvvm.mlir
mlir-translate example-nvvm.mlir        \
  --mlir-to-llvmir                      \ # Obtain the translated LLVM IR.
  -o example.ll
```

This compilation process expects all GPU code to live in a `gpu.module` and
expects all kernels to be `gpu.func` operations. Non-kernel functions, like
device library calls, may be defined using `func.func` or other non-GPU dialect
operations. This permits downstream systems to use these wrappers without
requiring them to use the GPU dialect's function operations, which might not include
information those systems want to have as intrinsic values on their functions.
Additionally, this allows for using `func.func` for device-side library functions
in `gpu.module`s.

### Default NVVM Compilation Pipeline: gpu-lower-to-nvvm-pipeline

The `gpu-lower-to-nvvm-pipeline` compilation pipeline serves as the default way
for NVVM target compilation within MLIR. This pipeline operates by lowering
primary dialects (arith, memref, scf, vector, gpu, and nvgpu) to NVVM target. It
begins by lowering GPU code region(s) to the specified NVVM compilation target
and subsequently handles the host code.

This pipeline specifically requires explicitly parallel IR and doesn't do GPU
parallelization. To enable parallelism, necessary transformations must be
applied before utilizing this pipeline.

It's designed to provide a generic solution for NVVM targets, generating NVVM
and LLVM dialect code compatible with `mlir-runner` or execution engine.

#### Example:

Here's a snippet illustrating the use of primary dialects, including arith,
within GPU code execution:

```
func.func @main() {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    gpu.launch
        blocks(%0, %1, %2) in (%3 = %c1, %4 = %c1, %5 = %c1)
        threads(%6, %7, %8) in (%9 = %c2, %10 = %c1, %11 = %c1) {
        gpu.printf "Hello from %d\n" %6 : index
        gpu.terminator
    }
    return
}
```

The `gpu-lower-to-nvvm` pipeline compiles this input code to NVVM format as
below. It provides customization options like specifying SM capability, PTX
version, and optimization level. Once compiled, the resulting IR is ready for
execution using `mlir-runner`. Alternatively, it can be translated into
LLVM, expanding its utility within the system.

```
mlir-opt example.mlir -gpu-lower-to-nvvm-pipeline = "cubin-chip=sm_90a cubin-features=+ptx80 opt-level=3"
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
