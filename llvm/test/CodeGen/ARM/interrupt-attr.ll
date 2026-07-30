; RUN: llc -mtriple=arm-none-none-eabi -mcpu=cortex-a15 -o - %s | FileCheck --check-prefix=CHECK-A %s
; RUN: llc -mtriple=thumb-none-none-eabi -mcpu=cortex-a15 -o - %s | FileCheck --check-prefix=CHECK-A-THUMB %s
; RUN: llc -mtriple=thumb-apple-none-macho -mcpu=cortex-m3 -o - %s | FileCheck --check-prefix=CHECK-M %s

declare arm_aapcscc void @bar()

@bigvar = global [16 x i32] zeroinitializer

define arm_aapcscc void @irq_fn() alignstack(8) "interrupt"="IRQ" {
  ; Must save all registers except banked sp and lr (we save lr anyway because
  ; we actually need it at the end to execute the return ourselves).

  ; Also need special function return setting pc and CPSR simultaneously.
; CHECK-A-LABEL: irq_fn:
; CHECK-A: push {r0, r1, r2, r3, r10, r11, r12, lr}
; CHECK-A: add r11, sp, #20
; CHECK-A-NOT: sub sp, sp, #{{[0-9]+}}
; CHECK-A: bfc sp, #0, #3
; CHECK-A: bl bar
; CHECK-A: sub sp, r11, #20
; CHECK-A: pop {r0, r1, r2, r3, r10, r11, r12, lr}
; CHECK-A: subs pc, lr, #4

; CHECK-A-THUMB-LABEL: irq_fn:
; CHECK-A-THUMB: push.w {r0, r1, r2, r3, r4, r7, r12, lr}
; CHECK-A-THUMB: add r7, sp, #20
; CHECK-A-THUMB: mov r4, sp
; CHECK-A-THUMB: bfc r4, #0, #3
; CHECK-A-THUMB: bl bar
; CHECK-A-THUMB: sub.w r4, r7,  #20
; CHECK-A-THUMB: mov sp, r4
; CHECK-A-THUMB: pop.w {r0, r1, r2, r3, r4, r7, r12, lr}
; CHECK-A-THUMB: subs pc, lr, #4

  ; Normal AAPCS function (r0-r3 pushed onto stack by hardware, lr set to
  ; appropriate sentinel so no special return needed).
; CHECK-M-LABEL: irq_fn:
; CHECK-M: push {r4, r6, r7, lr}
; CHECK-M: add r7, sp, #8
; CHECK-M: mov r4, sp
; CHECK-M: bfc r4, #0, #3
; CHECK-M: mov sp, r4
; CHECK-M: bl _bar
; CHECK-M: sub.w r4, r7, #8
; CHECK-M: mov sp, r4
; CHECK-M: pop {r4, r6, r7, pc}

  call arm_aapcscc void @bar()
  ret void
}

; We don't push/pop r12, as it is banked for FIQ
define arm_aapcscc void @fiq_fn() alignstack(8) "interrupt"="FIQ" {
; CHECK-A-LABEL: fiq_fn:
; CHECK-A: push {r0, r1, r2, r3, r4, r5, r6, r7, r11, lr}
  ; 32 to get past r0, r1, ..., r7
; CHECK-A: add r11, sp, #32
; CHECK-A: sub sp, sp, #{{[0-9]+}}
; CHECK-A: bfc sp, #0, #3
; [...]
  ; 32 must match above
; CHECK-A: sub sp, r11, #32
; CHECK-A: pop {r0, r1, r2, r3, r4, r5, r6, r7, r11, lr}
; CHECK-A: subs pc, lr, #4

; CHECK-A-THUMB-LABEL: fiq_fn:
; CHECK-M-LABEL: fiq_fn:
  %val = load volatile [16 x i32], ptr @bigvar
  store volatile [16 x i32] %val, ptr @bigvar
  ret void
}

define arm_aapcscc void @swi_fn() alignstack(8) "interrupt"="SWI" {
; CHECK-A-LABEL: swi_fn:
; CHECK-A: push {r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, lr}
; CHECK-A: add r11, sp, #44
; CHECK-A: sub sp, sp, #{{[0-9]+}}
; CHECK-A: bfc sp, #0, #3
; [...]
; CHECK-A: sub sp, r11, #44
; CHECK-A: pop {r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, lr}
; CHECK-A: subs pc, lr, #0

  %val = load volatile [16 x i32], ptr @bigvar
  store volatile [16 x i32] %val, ptr @bigvar
  ret void
}

define arm_aapcscc void @undef_fn() alignstack(8) "interrupt"="UNDEF" {
; CHECK-A-LABEL: undef_fn:
; CHECK-A: push {r0, r1, r2, r3, r10, r11, r12, lr}
; CHECK-A: add r11, sp, #20
; CHECK-A-NOT: sub sp, sp, #{{[0-9]+}}
; CHECK-A: bfc sp, #0, #3
; [...]
; CHECK-A: sub sp, r11, #20
; CHECK-A: pop {r0, r1, r2, r3, r10, r11, r12, lr}
; CHECK-A: subs pc, lr, #0

  call void @bar()
  ret void
}

define arm_aapcscc void @abort_fn() alignstack(8) "interrupt"="ABORT" {
; CHECK-A-LABEL: abort_fn:
; CHECK-A: push {r0, r1, r2, r3, r10, r11, r12, lr}
; CHECK-A: add r11, sp, #20
; CHECK-A-NOT: sub sp, sp, #{{[0-9]+}}
; CHECK-A: bfc sp, #0, #3
; [...]
; CHECK-A: sub sp, r11, #20
; CHECK-A: pop {r0, r1, r2, r3, r10, r11, r12, lr}
; CHECK-A: subs pc, lr, #4

  call void @bar()
  ret void
}

@var = global double 0.0

; We don't save VFP regs, since it would be a massive overhead in the general
; case.
define arm_aapcscc void @floating_fn() alignstack(8) "interrupt"="IRQ" {
; CHECK-A-LABEL: floating_fn:
; CHECK-A-NOT: vpush
; CHECK-A-NOT: vstr
; CHECK-A-NOT: vstm
; CHECK-A: vadd.f64 {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
  %lhs = load volatile double, ptr @var
  %rhs = load volatile double, ptr @var
  %sum = fadd double %lhs, %rhs
  store double %sum, ptr @var
  ret void
}
