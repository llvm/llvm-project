// Check that hooks with arguments work.
// RUN: tblgen -I $srcroot/include --gen-llvmc %s -o %t
// RUN: grep {Hook(const char\\* Arg0, const char\\* Arg1, const char\\* Arg2);} %t | count 1
// RUN: grep "/path" %t | count 1
// RUN: grep "VARIABLE" %t | count 1
// RUN: grep "/2path" %t | count 1

include "llvm/CompilerDriver/Common.td"

def dummy_tool : Tool<[
(cmd_line "$CALL(Hook, 'Arg1',   'Arg2', 'Arg3 Arg3Cont')/path arg1 $ENV(VARIABLE)/2path arg2 $INFILE"),
(in_language "dummy"),
(out_language "dummy")
]>;

def DummyGraph : CompilationGraph<[SimpleEdge<"root", "dummy_tool">]>;
