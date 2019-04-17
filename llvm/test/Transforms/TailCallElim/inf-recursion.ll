; RUN: opt < %s -tailcallelim -verify-dom-info -S | FileCheck %s

; Don't turn this into an infinite loop, this is probably the implementation
; of fabs and we expect the codegen to lower fabs.
; CHECK: @fabs(double %f)
; CHECK: call
; CHECK: ret

define double @fabs(double %f) {
entry:
        %tmp2 = call double @fabs( double %f )          ; <double> [#uses=1]
        ret double %tmp2
}

; Do turn other calls into infinite loops though.

; CHECK-LABEL: define double @foo(
; CHECK-NOT: call
; CHECK: }
define double @foo(double %f) {
        %t= call double @foo(double %f)
        ret double %t
}

; CHECK-LABEL: define float @fabsf(
; CHECK-NOT: call
; CHECK: }
define float @fabsf(float %f) {
        %t= call float @fabsf(float 2.0)
        ret float %t
}

declare x86_fp80 @fabsl(x86_fp80 %f)

; Don't crash while transforming a function with infinite recursion.
define i32 @PR22704(i1 %bool) {
entry:
  br i1 %bool, label %t, label %f

t:
  %call1 = call i32 @PR22704(i1 1)
  br label %return

f:
  %call = call i32 @PR22704(i1 1)
  br label %return

return:
  ret i32 0

; CHECK-LABEL: @PR22704(
; CHECK:       %bool.tr = phi i1 [ %bool, %entry ], [ true, %t ], [ true, %f ]
; CHECK:       br i1 %bool.tr, label %t, label %f
}
