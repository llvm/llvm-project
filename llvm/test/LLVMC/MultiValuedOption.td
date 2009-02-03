// Check that multivalued options work.
// The dummy tool and graph are required to silence warnings.
// RUN: tblgen -I $srcroot/include --gen-llvmc %s -o %t
// RUN: grep cl::multi_val(2) %t | count 1

include "llvm/CompilerDriver/Common.td"

def OptList : OptionList<[
    (prefix_list_option "foo", (multi_val 2)),
    (parameter_list_option "baz", (multi_val 2), (extern))]>;

def dummy_tool : Tool<[
(cmd_line "dummy_cmd"),
(in_language "dummy"),
(out_language "dummy"),
(actions (case
         (not_empty "foo"), (forward_as "foo", "bar"),
         (not_empty "baz"), (forward "baz")))
]>;

def DummyGraph : CompilationGraph<[SimpleEdge<"root", "dummy_tool">]>;
