// RUN: llvm-tblgen %s | FileCheck %s

// CHECK: class Y<list<string> Y:S = ?> {
// CHECK:   list<string> T1 = !listconcat(Y:S, ["foo"]);
// CHECK:   list<string> T2 = !listconcat(Y:S, !listconcat(["foo"], !listconcat(Y:S, ["bar", "baz"])));
// CHECK: }

// CHECK: def Z {
// CHECK:   list<string> T1 = ["fu", "foo"];
// CHECK:   list<string> T2 = ["fu", "foo", "fu", "bar", "baz"];
// CHECK: }

class Y<list<string> S> {
  list<string> T1 = !listconcat(S, ["foo"]);
  list<string> T2 = !listconcat(S, ["foo"], S, ["bar", "baz"]);
}

def Z : Y<["fu"]>;
