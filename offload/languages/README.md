# Offload Language Runtimes

This directory contains the CUDA and HIP runtime layer for LLVM offload. It
builds one shared library, `LLVMOffloadKernel`.

The installed language headers are the user-facing part of this directory. CUDA
programs include the CUDA header, HIP programs include the HIP header, and both
see normal language names such as `cudaMalloc` or `hipMalloc`.

Clang also depends on a small set of runtime entry points for kernel launch and
device image registration. These symbols are external because generated host
code calls them, but they are for compiler-generated code rather than for users
to call directly.

The rest of the runtime is internal. This includes device lookup, queues,
registered programs, registered kernels, error conversion, and per-thread
last-error storage.

Some source files are shared by CUDA and HIP. CMake compiles those files once
with `LANGUAGE=cuda` and once with `LANGUAGE=hip`. The language-name includes
rename the generic declarations and definitions to the CUDA or HIP spelling for
that object file.

Source files that do not depend on CUDA or HIP spelling are compiled once and
shared by both languages.

The `cuda/` and `hip/` directories are kept in case there is ever a need for
language-specific behavior. For now, the minimal differences between cuda and 
hip are taken care of in cuda/hip_runtime.h after the generic parts are included.
