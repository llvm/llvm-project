; RUN: llc < %s | FileCheck %s

target datalayout = "e-m:e-p:16:16-i32:16-i64:16-f32:16-f64:16-a:8-n8:16-S16"
target triple = "msp430-generic-generic"

@llvm.used = appending global [1 x i8*] [i8* bitcast (void ()* @ISR to i8*)], section "llvm.metadata"

; MSP430 EABI p. 3.9
; Interrupt functions must save all the registers that are used, even those
; that are normally considered callee-saved.

; To return from an interrupt function, the function must execute the special
; instruction RETI, which restores the SR register and branches to the PC where
; the interrupt occurred.

; CHECK:      .section	__interrupt_vector_2,"ax",@progbits
; CHECK-NEXT:	.short	ISR

@g = global float 0.0

define msp430_intrcc void @ISR() #0 {
entry:
; CHECK-LABEL: ISR:
; CHECK: push r4
; CHECK: push r5
; CHECK: push r6
; CHECK: push r7
; CHECK: push r8
; CHECK: push r9
; CHECK: push r10
; CHECK: push r11
; CHECK: push r12
; CHECK: push r13
; CHECK: push r14
; CHECK: push r15
  %t1 = load volatile float, float* @g
  %t2 = load volatile float, float* @g
  %t3 = load volatile float, float* @g
  %t4 = load volatile float, float* @g
  %t5 = load volatile float, float* @g
  %t6 = load volatile float, float* @g
  %t7 = load volatile float, float* @g
  store volatile float %t1, float* @g
  store volatile float %t2, float* @g
  store volatile float %t3, float* @g
  store volatile float %t4, float* @g
  store volatile float %t5, float* @g
  store volatile float %t6, float* @g
; CHECK: reti
  ret void
}

; Functions without 'interrupt' attribute don't get a vector section.
; CHECK-NOT: __interrupt_vector
; CHECK-LABEL: NMI:
; CHECK: reti
define msp430_intrcc void @NMI() #1 {
  ret void
}

attributes #0 = { noinline nounwind optnone "interrupt"="2" }
attributes #1 = { noinline nounwind optnone }
