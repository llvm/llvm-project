// Check that (one_or_more) and (zero_or_one) properties work.
// The dummy tool and graph are required to silence warnings.
// RUN: tblgen -I $srcroot/include --gen-llvmc %s -o %t
// RUN: grep cl::ZeroOrOne %t | count 1
// RUN: grep cl::OneOrMore %t | count 1

include "llvm/CompilerDriver/Common.td"

def OptList : OptionList<[
    (prefix_list_option "foo", (one_or_more)),
    (parameter_list_option "baz", (zero_or_one))]>;

def dummy_tool : Tool<[
(cmd_line "dummy_cmd"),
(in_language "dummy"),
(out_language "dummy"),
(actions (case
         (not_empty "foo"), (forward_as "foo", "bar"),
         (not_empty "baz"), (forward "baz")))
]>;

def DummyGraph : CompilationGraph<[SimpleEdge<"root", "dummy_tool">]>;
