This directory contains SPIR-V builtin functions used by
the LLVM module converted from SPIR-V by the SPIR-V/LLVM
converter. The SPIR-V consumers need to add these builtin
functions to their runtime library.

For OpenCL, most of the SPIR-V instructions are translated
to either LLVM instructions or OpenCL builtin function calls
by the converter. Therefore only a few SPIR-V instructions
need to be implemented in the runtime.