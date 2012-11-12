; RUN: opt < %s -instcombine -S | FileCheck %s

; CHECK: @t1
define zeroext i8 @t1(float %x, float %y) nounwind {
       %a = fcmp ueq float %x, %y             ; <i1> [#uses=1]
       %b = fcmp uno float %x, %y               ; <i1> [#uses=1]
       %c = or i1 %a, %b
; CHECK-NOT: fcmp uno
; CHECK: fcmp ueq
       %retval = zext i1 %c to i8
       ret i8 %retval
}

; CHECK: @t2
define zeroext i8 @t2(float %x, float %y) nounwind {
       %a = fcmp olt float %x, %y             ; <i1> [#uses=1]
       %b = fcmp oeq float %x, %y               ; <i1> [#uses=1]
; CHECK-NOT: fcmp olt
; CHECK-NOT: fcmp oeq
; CHECK: fcmp ole
       %c = or i1 %a, %b
       %retval = zext i1 %c to i8
       ret i8 %retval
}

; CHECK: @t3
define zeroext i8 @t3(float %x, float %y) nounwind {
       %a = fcmp ult float %x, %y             ; <i1> [#uses=1]
       %b = fcmp uge float %x, %y               ; <i1> [#uses=1]
       %c = or i1 %a, %b
       %retval = zext i1 %c to i8
; CHECK: ret i8 1
       ret i8 %retval
}

; CHECK: @t4
define zeroext i8 @t4(float %x, float %y) nounwind {
       %a = fcmp ult float %x, %y             ; <i1> [#uses=1]
       %b = fcmp ugt float %x, %y               ; <i1> [#uses=1]
       %c = or i1 %a, %b
; CHECK-NOT: fcmp ult
; CHECK-NOT: fcmp ugt
; CHECK: fcmp une
       %retval = zext i1 %c to i8
       ret i8 %retval
}

; CHECK: @t5
define zeroext i8 @t5(float %x, float %y) nounwind {
       %a = fcmp olt float %x, %y             ; <i1> [#uses=1]
       %b = fcmp oge float %x, %y               ; <i1> [#uses=1]
       %c = or i1 %a, %b
; CHECK-NOT: fcmp olt
; CHECK-NOT: fcmp oge
; CHECK: fcmp ord
       %retval = zext i1 %c to i8
       ret i8 %retval
}
