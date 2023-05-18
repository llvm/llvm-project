// Check that not specifying a valid condition results in error

// RUN: not llvm-tblgen %s 2>&1 | FileCheck %s
// XFAIL: vg_leak

class C<int x> {
  string s  = !cond(!lt(x,0) : "negative", !gt(x,0) : "positive");
}

def Zero : C<0>;
//CHECK: error: Zero does not have any true condition in:!cond(0: "negative", 0: "positive")
