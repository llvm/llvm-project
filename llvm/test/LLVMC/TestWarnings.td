// Check that the compiler warns about unused options.
// This should fail because the output is printed on stderr.
// RUN: ignore tblgen -I $srcroot/include --gen-llvmc %s |& grep "option '-Wall' has no effect!"

include "llvm/CompilerDriver/Common.td"

def OptList : OptionList<[(switch_option "Wall", (extern))]>;
