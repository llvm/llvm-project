; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

; Function Attrs: noinline nounwind optnone
define dso_local i64 @bitcastd2l(double %x) {
; CHECK-LABEL: bitcastd2l:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %r = bitcast double %x to i64
  ret i64 %r
}

; Function Attrs: noinline nounwind optnone
define dso_local double @bitcastl2d(i64 %x) {
; CHECK-LABEL: bitcastl2d:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %r = bitcast i64 %x to double
  ret double %r
}

; Function Attrs: noinline nounwind optnone
define dso_local float @bitcastw2f(i32 %x) {
; CHECK-LABEL: bitcastw2f:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    sll %s0, %s0, 32
; CHECK-NEXT:    or %s11, 0, %s9
  %r = bitcast i32 %x to float
  ret float %r
}

; Function Attrs: noinline nounwind optnone
define dso_local i32 @bitcastf2w(float %x) {
; CHECK-LABEL: bitcastf2w:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    sra.l %s0, %s0, 32
; CHECK-NEXT:    or %s11, 0, %s9
  %r = bitcast float %x to i32
  ret i32 %r
}
