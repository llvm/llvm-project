// RUN: llvm-tblgen %s | FileCheck %s

// CHECK: class X {
// CHECK:   list<string> T0 = ["foo", "bar"];
// CHECK:   list<string> T1 = ["foo", "bar"];
// CHECK:   list<string> T2 = ["bar"];
// CHECK:   list<string> T3 = ["foo"];
// CHECK:   list<string> T4 = [];
// CHECK: }

class X {
  list<string> T0 = !listremove(["foo", "bar"], []);
  list<string> T1 = !listremove(["foo", "bar"], ["baz"]);
  list<string> T2 = !listremove(["foo", "bar"], ["foo"]);
  list<string> T3 = !listremove(["foo", "bar"], ["bar", "bar"]);
  list<string> T4 = !listremove(["foo", "bar"], ["bar", "foo"]);
}
