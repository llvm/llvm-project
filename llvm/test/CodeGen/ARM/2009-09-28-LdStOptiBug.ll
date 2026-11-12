; RUN: llc < %s -mtriple=armv5-unknown-linux-gnueabi -mcpu=arm10tdmi | FileCheck %s
; PR4687

%0 = type { double, double }

define void @foo(ptr noalias nocapture sret(%0) %agg.result, double %x.0, double %y.0) nounwind {
; CHECK-LABEL: foo:
; CHECK: bl __aeabi_dadd
; CHECK-NOT: strd
; CHECK: mov
  %x76 = fmul double %y.0, 0.000000e+00           ; <double> [#uses=1]
  %x77 = fadd double %y.0, 0.000000e+00           ; <double> [#uses=1]
  %tmpr = fadd double %x.0, %x76                  ; <double> [#uses=1]
  %agg.result.0 = getelementptr %0, ptr %agg.result, i32 0, i32 0 ; <ptr> [#uses=1]
  store double %tmpr, ptr %agg.result.0, align 8
  %agg.result.1 = getelementptr %0, ptr %agg.result, i32 0, i32 1 ; <ptr> [#uses=1]
  store double %x77, ptr %agg.result.1, align 8
  ret void
}
