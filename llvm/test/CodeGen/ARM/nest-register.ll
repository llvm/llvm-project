; RUN: llc -mtriple=arm-eabi %s -o - | FileCheck %s

; Tests that the 'nest' parameter attribute causes the relevant parameter to be
; passed in the right register.

define i8* @nest_receiver(i8* nest %arg) nounwind {
; CHECK-LABEL: nest_receiver:
; CHECK: @ %bb.0:
; CHECK-NEXT: mov r0, r12
; CHECK-NEXT: mov pc, lr
        ret i8* %arg
}

define i8* @nest_caller(i8* %arg) nounwind {
; CHECK-LABEL: nest_caller:
; CHECK: mov r12, r0
; CHECK-NEXT: bl nest_receiver
; CHECK: mov pc, lr
        %result = call i8* @nest_receiver(i8* nest %arg)
        ret i8* %result
}
