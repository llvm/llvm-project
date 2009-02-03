// Check that multiple compilation graphs are allowed.
// RUN: tblgen -I $srcroot/include --gen-llvmc %s

include "llvm/CompilerDriver/Common.td"

def Graph1 : CompilationGraph<[]>;
def Graph2 : CompilationGraph<[]>;
def Graph3 : CompilationGraph<[]>;
