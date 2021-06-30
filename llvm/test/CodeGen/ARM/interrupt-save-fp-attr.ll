; RUN: llc -mtriple=arm-none-none-eabi -mcpu=cortex-a15 -o - %s | FileCheck --check-prefix=CHECK-A %s
; RUN: llc -mtriple=thumb-none-none-eabi -mcpu=cortex-a15 -o - %s | FileCheck --check-prefix=CHECK-A-THUMB %s
; RUN: llc -mtriple=thumb-apple-none-macho -mcpu=cortex-m4 -o - %s | FileCheck --check-prefix=CHECK-M %s
; RUN: llc -mtriple=thumbv7em-ti-none-eabihf -mcpu=cortex-m4 -o - %s | FileCheck --check-prefix=CHECK-M %s
; RUN: llc -mtriple=thumbv7r5-ti-none-eabihf -mcpu=cortex-r5 -o - %s | FileCheck --check-prefix=CHECK-R-THUMB %s
; RUN: llc -mtriple=armv7r5-ti-none-eabihf -mcpu=cortex-r5 -o - %s | FileCheck --check-prefix=CHECK-R %s

declare arm_aapcscc void @bar()

@bigvar = global [16 x i32] zeroinitializer

define arm_aapcscc void @irq_fn() alignstack(8) "interrupt"="IRQ" "save-fp"{
  ; Must save all registers except banked sp and lr (we save lr anyway because
  ; we actually need it at the end to execute the return ourselves).

  ; Also need special function return setting pc and CPSR simultaneously.
; CHECK-A-LABEL: irq_fn:
; CHECK-A: push {r0, r1, r2, r3, r10, r11, r12, lr}
; CHECK-A: add r11, sp, #20
; CHECK-A-NOT: sub sp, sp, #{{[0-9]+}}
; CHECK-A: .vsave {d0, d1, d2, d3, d4, d5, d6, d7}
; CHECK-A: vpush {d0, d1, d2, d3, d4, d5, d6, d7}
; CHECK-A: .vsave {d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31}
; CHECK-A: vpush {d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31}
; CHECK-A: bl bar
; CHECK-A: sub sp, r11, #212
; CHECK-A: vpop {d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31}
; CHECK-A: vpop {d0, d1, d2, d3, d4, d5, d6, d7}
; CHECK-A: pop {r0, r1, r2, r3, r10, r11, r12, lr}
; CHECK-A: subs pc, lr, #4

; CHECK-A-THUMB-LABEL: irq_fn:
; CHECK-A-THUMB: push.w {r0, r1, r2, r3, r4, r7, r12, lr}
; CHECK-A-THUMB: add r7, sp, #20
; CHECK-A-THUMB: .vsave {d0, d1, d2, d3, d4, d5, d6, d7}
; CHECK-A-THUMB: vpush {d0, d1, d2, d3, d4, d5, d6, d7}
; CHECK-A-THUMB: .vsave {d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31}
; CHECK-A-THUMB: vpush {d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31}
; CHECK-A-THUMB: mov r4, sp
; CHECK-A-THUMB: bfc r4, #0, #3
; CHECK-A-THUMB: bl bar
; CHECK-A-THUMB: sub.w r4, r7,  #212
; CHECK-A-THUMB: mov sp, r4
; CHECK-A-THUMB: vpop {d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31}
; CHECK-A-THUMB: vpop {d0, d1, d2, d3, d4, d5, d6, d7}
; CHECK-A-THUMB: pop.w {r0, r1, r2, r3, r4, r7, r12, lr}
; CHECK-A-THUMB: subs pc, lr, #4

; CHECK-R-LABEL: irq_fn:
; CHECK-R: push {r0, r1, r2, r3, r10, r11, r12, lr}
; CHECK-R: add r11, sp, #20
; CHECK-R-NOT: sub sp, sp, #{{[0-9]+}}
; CHECK-R: bfc sp, #0, #3
; CHECK-R: .vsave {d0, d1, d2, d3, d4, d5, d6, d7}
; CHECK-R: vpush {d0, d1, d2, d3, d4, d5, d6, d7}
; CHECK-R: bl bar
; CHECK-R: sub sp, r11, #84
; CHECK-R: vpop {d0, d1, d2, d3, d4, d5, d6, d7}
; CHECK-R: pop {r0, r1, r2, r3, r10, r11, r12, lr}
; CHECK-R: subs pc, lr, #4

; CHECK-R-THUMB-LABEL: irq_fn:
; CHECK-R-THUMB: push.w {r0, r1, r2, r3, r4, r7, r12, lr}
; CHECK-R-THUMB: add r7, sp, #20
; CHECK-R-THUMB: mov r4, sp
; CHECK-R-THUMB: bfc r4, #0, #3
; CHECK-R-THUMB: .vsave {d0, d1, d2, d3, d4, d5, d6, d7}
; CHECK-R-THUMB: vpush {d0, d1, d2, d3, d4, d5, d6, d7}
; CHECK-R-THUMB: bl bar
; CHECK-R-THUMB: vpop {d0, d1, d2, d3, d4, d5, d6, d7}
; CHECK-R-THUMB: sub.w r4, r7,  #84
; CHECK-R-THUMB: mov sp, r4
; CHECK-R-THUMB: pop.w {r0, r1, r2, r3, r4, r7, r12, lr}
; CHECK-R-THUMB: subs pc, lr, #4

  ; Normal AAPCS function (r0-r3 pushed onto stack by hardware, lr set to
  ; appropriate sentinel so no special return needed).
; CHECK-M-LABEL: irq_fn:
; CHECK-M: push {r4, r6, r7, lr}
; CHECK-M: add r7, sp, #8
; CHECK-M: vpush {d0, d1, d2, d3, d4, d5, d6, d7}
; CHECK-M: mov r4, sp
; CHECK-M: bfc r4, #0, #3
; CHECK-M: mov sp, r4
; CHECK-M: bl {{_?}}bar
; CHECK-M: sub.w r4, r7, #72
; CHECK-M: mov sp, r4
; CHECK-M: vpop	{d0, d1, d2, d3, d4, d5, d6, d7}
; CHECK-M: pop {r4, r6, r7, pc}

  call arm_aapcscc void @bar()
  ret void
}

; We don't push/pop r12, as it is banked for FIQ
define arm_aapcscc void @fiq_fn() alignstack(8) "interrupt"="FIQ" "save-fp" {
; CHECK-A-LABEL: fiq_fn:
; CHECK-A: push {r0, r1, r2, r3, r4, r5, r6, r7, r11, lr}
  ; 32 to get past r0, r1, ..., r7
; CHECK-A: .setfp	r11, sp, #32
; CHECK-A: add	r11, sp, #32
; CHECK-A: sub	sp, sp, #16
; CHECK-A: bfc	sp, #0, #3
; [...]
  ; 32 must match above
; CHECK-A: sub sp, r11, #32
; CHECK-A: pop {r0, r1, r2, r3, r4, r5, r6, r7, r11, lr}
; CHECK-A: subs pc, lr, #4

; CHECK-A-THUMB-LABEL: fiq_fn:
; CHECK-M-LABEL: fiq_fn:
  %val = load volatile [16 x i32], [16 x i32]* @bigvar
  store volatile [16 x i32] %val, [16 x i32]* @bigvar
  ret void
}

define arm_aapcscc void @swi_fn() alignstack(8) "interrupt"="SWI" "save-fp" {
; CHECK-A-LABEL: swi_fn:
; CHECK-A: push {r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, lr}
; CHECK-A: add r11, sp, #44
; CHECK-A: sub sp, sp, #{{[0-9]+}}
; CHECK-A: bfc sp, #0, #3
; [...]
; CHECK-A: sub sp, r11, #44
; CHECK-A: pop {r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, lr}
; CHECK-A: subs pc, lr, #0

  %val = load volatile [16 x i32], [16 x i32]* @bigvar
  store volatile [16 x i32] %val, [16 x i32]* @bigvar
  ret void
}

define arm_aapcscc void @undef_fn() alignstack(8) "interrupt"="UNDEF" "save-fp" {
; CHECK-A-LABEL: undef_fn:
; CHECK-A: push {r0, r1, r2, r3, r10, r11, r12, lr}
; CHECK-A: add r11, sp, #20
; CHECK-A-NOT: sub sp, sp, #{{[0-9]+}}
; CHECK-A: .vsave	{d0, d1, d2, d3, d4, d5, d6, d7}
; CHECK-A: vpush	{d0, d1, d2, d3, d4, d5, d6, d7}
; CHECK-A: .vsave	{d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31}
; CHECK-A: vpush	{d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31}
; [...]
; CHECK-A: sub sp, r11, #212
; CHECK-A: vpop	{d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31}
; CHECK-A: vpop	{d0, d1, d2, d3, d4, d5, d6, d7}
; CHECK-A: pop {r0, r1, r2, r3, r10, r11, r12, lr}
; CHECK-A: subs pc, lr, #0

  call void @bar()
  ret void
}

define arm_aapcscc void @abort_fn() alignstack(8) "interrupt"="ABORT" "save-fp" {
; CHECK-A-LABEL: abort_fn:
; CHECK-A: push {r0, r1, r2, r3, r10, r11, r12, lr}
; CHECK-A: add r11, sp, #20
; CHECK-A-NOT: sub sp, sp, #{{[0-9]+}}
; CHECK-A: .vsave	{d0, d1, d2, d3, d4, d5, d6, d7}
; CHECK-A: vpush	{d0, d1, d2, d3, d4, d5, d6, d7}
; [...]
; CHECK-A: sub sp, r11, #212
; CHECK-A: vpop	{d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31}
; CHECK-A: vpop	{d0, d1, d2, d3, d4, d5, d6, d7}
; CHECK-A: pop {r0, r1, r2, r3, r10, r11, r12, lr}
; CHECK-A: subs pc, lr, #4

  call void @bar()
  ret void
}

@var = global double 0.0
