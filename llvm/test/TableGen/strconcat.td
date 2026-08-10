// RUN: llvm-tblgen %s | FileCheck %s

// CHECK: class Y<string Y:S = ?> {
// CHECK:   string T = !strconcat(Y:S, "foo");
// CHECK:   string T2 = !strconcat(Y:S, !strconcat("foo", !strconcat(Y:S, "bar")));
// CHECK:   string S = "foobar";
// CHECK: }

// CHECK: def Z {
// CHECK:   string T = "fufoo";
// CHECK:   string T2 = "fufoofubar";
// CHECK:   string S = "foobar";
// CHECK: }

class Y<string S> {
  string T = !strconcat(S, "foo");
  // More than two arguments is equivalent to nested calls
  string T2 = !strconcat(S, "foo", S, "bar");

  // String values concatenate lexically, as in C.
  string S = "foo" "bar";
}

def Z : Y<"fu">;
