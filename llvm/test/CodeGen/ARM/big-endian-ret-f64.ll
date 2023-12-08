; RUN: llc -mtriple=armebv7a-eabi %s -O0 -o - | FileCheck %s
; RUN: llc -mtriple=armebv8a-eabi %s -O0 -o - | FileCheck %s

define double @fn() {
; CHECK-LABEL: fn
; CHECK: ldr r0, [sp]
; CHECK: ldr r1, [sp, #4]
  %r = alloca double, align 8
  %1 = load double, ptr %r, align 8
  ret double %1
}

