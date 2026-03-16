; RUN: opt -S -passes=instcombine < %s | FileCheck %s

declare i1 @llvm.is.fpclass.f64(double, i32)

define i1 @test_frem_is_fpclass_inf(double %a, double %b) {
; CHECK-LABEL: @test_frem_is_fpclass_inf(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i1 false
;
entry:
  %rem = frem double %a, %b
  ; 516 checks for +inf (512) or -inf (4)
  %is.inf = call i1 @llvm.is.fpclass.f64(double %rem, i32 516)
  ret i1 %is.inf
}

define i1 @test_frem_fcmp_inf(double %a, double %b) {
; CHECK-LABEL: @test_frem_fcmp_inf(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i1 false
;
entry:
  %rem = frem double %a, %b
  %is.inf = fcmp oeq double %rem, 0x7FF0000000000000
  ret i1 %is.inf
}
