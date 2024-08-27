; RUN: llc -mtriple=arm-eabi -mattr=+neon -float-abi=soft %s -o - | FileCheck %s

; CHECK: function1
; CHECK-NOT: vmov
define double @function1(double %a, double %b, double %c, double %d, double %e, double %f) nounwind noinline ssp {
; CHECK-LABEL: function1:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    .save {r4, r5, r11, lr}
; CHECK-NEXT:    push {r4, r5, r11, lr}
; CHECK-NEXT:    vldr d16, [sp, #40]
; CHECK-NEXT:    vldr d17, [sp, #32]
; CHECK-NEXT:    vmov r12, lr, d16
; CHECK-NEXT:    vldr d16, [sp, #16]
; CHECK-NEXT:    vmov r4, r5, d17
; CHECK-NEXT:    vldr d17, [sp, #24]
; CHECK-NEXT:    str r3, [sp, #36]
; CHECK-NEXT:    str r2, [sp, #32]
; CHECK-NEXT:    str r1, [sp, #44]
; CHECK-NEXT:    str r0, [sp, #40]
; CHECK-NEXT:    vstr d17, [sp, #16]
; CHECK-NEXT:    vstr d16, [sp, #24]
; CHECK-NEXT:    mov r0, r12
; CHECK-NEXT:    mov r1, lr
; CHECK-NEXT:    mov r2, r4
; CHECK-NEXT:    mov r3, r5
; CHECK-NEXT:    pop {r4, r5, r11, lr}
; CHECK-NEXT:    b function2
entry:
  %call = tail call double @function2(double %f, double %e, double %d, double %c, double %b, double %a) nounwind
  ret double %call
}

declare double @function2(double, double, double, double, double, double)
