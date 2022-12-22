; RUN: llc -mtriple=arm-eabi %s -o - | FileCheck %s

; Tests that the 'nest' parameter attribute causes the relevant parameter to be
; passed in the right register.

define ptr @nest_receiver(ptr nest %arg) nounwind {
; CHECK-LABEL: nest_receiver:
; CHECK: @ %bb.0:
; CHECK-NEXT: mov r0, r12
; CHECK-NEXT: mov pc, lr
        ret ptr %arg
}

define ptr @nest_caller(ptr %arg) nounwind {
; CHECK-LABEL: nest_caller:
; CHECK: mov r12, r0
; CHECK-NEXT: bl nest_receiver
; CHECK: mov pc, lr
        %result = call ptr @nest_receiver(ptr nest %arg)
        ret ptr %result
}
