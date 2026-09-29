// RUN: llvm-tblgen %s | FileCheck %s

// CHECK: class X<list<int> X:a = ?, list<int> X:b = ?, list<int> X:c = ?> {
// CHECK:   list<int> x = !listconcat(!listconcat(X:a, X:b), !listconcat(X:b, X:c));
// CHECK: }

// CHECK: class Y<list<string> Y:S = ?> {
// CHECK:   list<string> T1 = !listconcat(Y:S, ["foo"]);
// CHECK:   list<string> T2 = !listconcat(Y:S, !listconcat(["foo"], !listconcat(Y:S, ["bar", "baz"])));
// CHECK: }

// CHECK: def A0 {
// CHECK:   list<int> lst = [4];
// CHECK: }

// CHECK: def A1 {
// CHECK:   list<int> lst = [];
// CHECK: }

// CHECK: def DX {
// CHECK:   list<int> x = [0, 1, 1, 2]
// CHECK: }

// CHECK: def Z {
// CHECK:   list<string> T1 = ["fu", "foo"];
// CHECK:   list<string> T2 = ["fu", "foo", "fu", "bar", "baz"];
// CHECK: }

class A<bit x> {
  // The empty lists type-check without issues.
  list<int> lst = !listconcat([], !if(x, [], [4]));
}

def A0 : A<0>;
def A1 : A<1>;

class X<list<int> a, list<int> b, list<int> c> {
    list<int> x = !listconcat(!listconcat(a, b), !listconcat(b, c));
}

class Y<list<string> S> {
  list<string> T1 = !listconcat(S, ["foo"]);
  list<string> T2 = !listconcat(S, ["foo"], S, ["bar", "baz"]);
}

def DX : X<[0], [1], [2]>;

def Z : Y<["fu"]>;
