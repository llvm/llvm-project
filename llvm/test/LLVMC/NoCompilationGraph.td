// Check that the compilation graph is not required.
// RUN: tblgen -I $srcroot/include --gen-llvmc %s

include "llvm/CompilerDriver/Common.td"
