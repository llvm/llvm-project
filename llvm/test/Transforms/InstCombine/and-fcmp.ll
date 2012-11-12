; RUN: opt < %s -instcombine -S | FileCheck %s

define zeroext i8 @t1(float %x, float %y) nounwind {
       %a = fcmp ueq float %x, %y
       %b = fcmp ord float %x, %y
       %c = and i1 %a, %b
       %retval = zext i1 %c to i8
       ret i8 %retval
; CHECK: t1
; CHECK: fcmp oeq float %x, %y
; CHECK-NOT: fcmp ueq float %x, %y
; CHECK-NOT: fcmp ord float %x, %y
; CHECK-NOT: and
}

define zeroext i8 @t2(float %x, float %y) nounwind {
       %a = fcmp olt float %x, %y
       %b = fcmp ord float %x, %y
       %c = and i1 %a, %b
       %retval = zext i1 %c to i8
       ret i8 %retval
; CHECK: t2
; CHECK: fcmp olt float %x, %y
; CHECK-NOT: fcmp ord float %x, %y
; CHECK-NOT: and
}

define zeroext i8 @t3(float %x, float %y) nounwind {
       %a = fcmp oge float %x, %y
       %b = fcmp uno float %x, %y
       %c = and i1 %a, %b
       %retval = zext i1 %c to i8
       ret i8 %retval
; CHECK: t3
; CHECK: ret i8 0
}

define zeroext i8 @t4(float %x, float %y) nounwind {
       %a = fcmp one float %y, %x
       %b = fcmp ord float %x, %y
       %c = and i1 %a, %b
       %retval = zext i1 %c to i8
       ret i8 %retval
; CHECK: t4
; CHECK: fcmp one float %y, %x
; CHECK-NOT: fcmp ord float %x, %y
; CHECK-NOT: and
}

define zeroext i8 @t5(float %x, float %y) nounwind {
       %a = fcmp ord float %x, %y
       %b = fcmp uno float %x, %y
       %c = and i1 %a, %b
       %retval = zext i1 %c to i8
       ret i8 %retval
; CHECK: t5
; CHECK: ret i8 0
}

define zeroext i8 @t6(float %x, float %y) nounwind {
       %a = fcmp uno float %x, %y
       %b = fcmp ord float %x, %y
       %c = and i1 %a, %b
       %retval = zext i1 %c to i8
       ret i8 %retval
; CHECK: t6
; CHECK: ret i8 0
}

define zeroext i8 @t7(float %x, float %y) nounwind {
       %a = fcmp uno float %x, %y
       %b = fcmp ult float %x, %y
       %c = and i1 %a, %b
       %retval = zext i1 %c to i8
       ret i8 %retval
; CHECK: t7
; CHECK: fcmp uno
; CHECK-NOT: fcmp ult
}
