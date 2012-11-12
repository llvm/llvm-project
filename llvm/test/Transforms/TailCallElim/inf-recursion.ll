; RUN: opt < %s -tailcallelim -S | FileCheck %s

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

; CHECK: define double @foo
; CHECK-NOT: call
; CHECK: }
define double @foo(double %f) {
        %t= call double @foo(double %f)
        ret double %t
}

; CHECK: define float @fabsf
; CHECK-NOT: call
; CHECK: }
define float @fabsf(float %f) {
        %t= call float @fabsf(float 2.0)
        ret float %t
}

declare x86_fp80 @fabsl(x86_fp80 %f)
