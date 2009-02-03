// Check that tools without associated actions are accepted.
// RUN: tblgen -I $srcroot/include --gen-llvmc %s | grep dummy_tool

include "llvm/CompilerDriver/Common.td"

def dummy_tool : Tool<[
(cmd_line "dummy_cmd"),
(in_language "dummy"),
(out_language "dummy")
]>;

def DummyGraph : CompilationGraph<[SimpleEdge<"root", "dummy_tool">]>;
