// RUN: not llvm-tblgen %s 2>&1 | FileCheck %s
// XFAIL: vg_leak

class A<int dummy> {}
class B<int dummy> : A<dummy> {}
class C<int dummy> : A<dummy> {}

// CHECK: Field 'x' of type 'C' is incompatible with value '{{.*}}' of type 'A'
class X<int cc, B b, C c> {
  C x = !cond(cc: b, 1 : c);
}
