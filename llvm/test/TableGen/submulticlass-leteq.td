// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak
// CHECK:      def X0 { // C
// CHECK-NEXT:   bit x = 1;
// CHECK-NEXT: }
// CHECK-NEXT: def X1 { // C
// CHECK-NEXT:   bit x = 0;
// CHECK-NEXT: }
class C {
  bit x;
}
multiclass M0<bits<8> Val> {
  let x = !eq(Val, !cast<bits<8>>(-1)) in def NAME : C;
}
multiclass M1<bits<8> Val> {
  let x = !eq(Val, -1) in def NAME : C;
}
multiclass M2_0 : M0<-1>;
multiclass M2_1 : M1<-1>;
defm X0 : M2_0;
defm X1 : M2_1;
