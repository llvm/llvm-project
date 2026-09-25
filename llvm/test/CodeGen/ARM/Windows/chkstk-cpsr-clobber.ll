; RUN: llc -O2 -mtriple=thumbv7-windows %s -o - | FileCheck %s

; Ensure that __chkstk correctly marks CPSR as clobbered so that conditional
; operations after dynamic stack allocation re-evaluate condition flags.

define arm_aapcs_vfpcc void @chkstk_cpsr_clobber(i32 %n) {
entry:
  %cmp = icmp sgt i32 %n, 0
  %max = select i1 %cmp, i32 %n, i32 1
  %vla = alloca i8, i32 %max
  br i1 %cmp, label %then, label %exit

then:
  store volatile i8 1, ptr %vla
  ret void

exit:
  ret void
}

; CHECK-LABEL: chkstk_cpsr_clobber:
; CHECK:         bl __chkstk
; CHECK:         sub.w sp, sp, r4
; CHECK:         cmp r0, #0
; CHECK:         it gt
