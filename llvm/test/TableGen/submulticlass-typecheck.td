// RUN: not llvm-tblgen %s 2>&1 | FileCheck %s
// XFAIL: vg_leak
// CHECK: {{.*}}:11:28: error: Value specified for template argument 'B::op' is of type bits<4>; expected type bits<8>: C::op
// CHECK-NEXT: multiclass C<bits<4> op> : B<op>;
class A<bits<8> op> {
  bits<8> f = op;
}
multiclass B<bits<8> op> {
  def : A<op>;
}
multiclass C<bits<4> op> : B<op>;
defm D : C<0>;
