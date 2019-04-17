; RUN: opt < %s -instsimplify -S | FileCheck %s

declare double @llvm.powi.f64(double, i32) nounwind readonly
declare i32 @llvm.bswap.i32(i32)

; A
define i32 @test_bswap(i32 %a) nounwind {
; CHECK-LABEL: @test_bswap(
; CHECK-NEXT:    ret i32 %a
;
  %tmp2 = tail call i32 @llvm.bswap.i32( i32 %a )
  %tmp4 = tail call i32 @llvm.bswap.i32( i32 %tmp2 )
  ret i32 %tmp4
}

define void @powi(double %V, double *%P) {
  %B = tail call double @llvm.powi.f64(double %V, i32 0) nounwind
  store volatile double %B, double* %P

  %C = tail call double @llvm.powi.f64(double %V, i32 1) nounwind
  store volatile double %C, double* %P

  ret void
; CHECK-LABEL: @powi(
; CHECK: store volatile double 1.0
; CHECK: store volatile double %V
}
