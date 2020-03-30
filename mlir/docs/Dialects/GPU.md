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

## Operations

[include "Dialects/GPUOps.md"]
