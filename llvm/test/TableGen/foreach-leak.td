// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

// CHECK: --- Classes ---
// CHECK: list<int> ret = !foreach(a,

// CHECK: --- Defs ---

// CHECK: def C0 {
// CHECK{LITERAL}: list<list<int>> ret = [[1, 2, 3], [1, 2, 3]];
// CHECK: }

// The variable name 'a' is used both in the "inner" and in the "outer" foreach.
// The test ensure that the inner declaration of 'a' properly shadows the outer
// one.
class A<list<int> lst> {
  list<int> ret = !foreach(a, lst, !add(a, 1));
}

class B<list<int> lst1, list<int> lst2> {
  list<list<int>> ret = !foreach(a, lst1, A<lst2>.ret);
}

class C<list<int> lst2> {
  list<list<int>> ret = B<[0, 1], lst2>.ret;
}

def C0 : C<[0, 1, 2]>;
