// RUN: llvm-tblgen %s | grep fufoo
// XFAIL: vg_leak

class Y<string S> {
  string T = !strconcat(S, "foo");

  // String values concatenate lexically, as in C.
  string S = "foo" "bar";
}

def Z : Y<"fu">;
