StreamExecutor
==============

StreamExecutor is a wrapper around CUDA and OpenCL (host-side) programming
models (runtimes). This abstraction cleanly permits host code to target either
CUDA or OpenCL devices with identically-functioning data parallel kernels. It
manages the execution of concurrent work targeting the accelerator, similar to a
host-side Executor.

This version of StreamExecutor can be built either as a sub-project of the LLVM
project or as a standalone project depending on LLVM as an external package.
