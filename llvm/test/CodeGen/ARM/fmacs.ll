; RUN: llc -mtriple=arm-eabi -mattr=+vfp2 %s -o - | FileCheck %s -check-prefix=VFP2
; RUN: llc -mtriple=arm-eabi -mattr=+neon %s -o - | FileCheck %s -check-prefix=NEON
; RUN: llc -mtriple=arm-eabi -mcpu=cortex-a8 %s -o - | FileCheck %s -check-prefix=A8
; RUN: llc -mtriple=arm-eabi -mcpu=cortex-a9 %s -o - | FileCheck %s -check-prefix=A9
; RUN: llc -mtriple=arm-linux-gnueabi -mcpu=cortex-a9 -float-abi=hard %s -o - | FileCheck %s -check-prefix=HARD
; RUN: llc -mtriple=arm-linux-gnueabi -mcpu=cortex-m4 -float-abi=hard %s -o - | FileCheck %s -check-prefix=VMLA
; RUN: llc -mtriple=arm-linux-gnueabi -mcpu=cortex-m33 -float-abi=hard %s -o - | FileCheck %s -check-prefix=VMLA

define float @t1(float %acc, float %a, float %b) {
entry:
; VFP2-LABEL: t1:
; VFP2: vmla.f32

; NEON-LABEL: t1:
; NEON: vmla.f32

; A8-LABEL: t1:
; A8: vmul.f32
; A8: vadd.f32

; VMLA-LABEL: t1:
; VMLA:       vmul.f32
; VMLA-NEXT:  vadd.f32

  %0 = fmul float %a, %b
  %1 = fadd float %acc, %0
	ret float %1
}

define float @vmla_minsize(float %acc, float %a, float %b) #0 {
entry:
; VMLA-LABEL: vmla_minsize:
; VMLA:       vmla.f32  s0, s1, s2
; VMLA-NEXT:  bx  lr

  %0 = fmul float %a, %b
  %1 = fadd float %acc, %0
	ret float %1
}

define double @t2(double %acc, double %a, double %b) {
entry:
; VFP2-LABEL: t2:
; VFP2: vmla.f64

; NEON-LABEL: t2:
; NEON: vmla.f64

; A8-LABEL: t2:
; A8: vmul.f64
; A8: vadd.f64
  %0 = fmul double %a, %b
  %1 = fadd double %acc, %0
	ret double %1
}

define float @t3(float %acc, float %a, float %b) {
entry:
; VFP2-LABEL: t3:
; VFP2: vmla.f32

; NEON-LABEL: t3:
; NEON: vmla.f32

; A8-LABEL: t3:
; A8: vmul.f32
; A8: vadd.f32
  %0 = fmul float %a, %b
  %1 = fadd float %0, %acc
	ret float %1
}

; It's possible to make use of fp vmla / vmls on Cortex-A9.
; rdar://8659675
define void @t4(float %acc1, float %a, float %b, float %acc2, float %c, ptr %P1, ptr %P2) {
entry:
; A8-LABEL: t4:
; A8: vmul.f32
; A8: vmul.f32
; A8: vadd.f32
; A8: vadd.f32

; Two vmla with now RAW hazard
; A9-LABEL: t4:
; A9: vmla.f32
; A9: vmla.f32

; HARD-LABEL: t4:
; HARD: vmla.f32 s0, s1, s2
; HARD: vmla.f32 s3, s1, s4
  %0 = fmul float %a, %b
  %1 = fadd float %acc1, %0
  %2 = fmul float %a, %c
  %3 = fadd float %acc2, %2
  store float %1, ptr %P1
  store float %3, ptr %P2
  ret void
}

define float @t5(float %a, float %b, float %c, float %d, float %e) {
entry:
; A8-LABEL: t5:
; A8: vmul.f32
; A8: vmul.f32
; A8: vadd.f32
; A8: vadd.f32

; A9-LABEL: t5:
; A9: vmla.f32
; A9: vmul.f32
; A9: vadd.f32

; HARD-LABEL: t5:
; HARD: vmla.f32 s4, s0, s1
; HARD: vmul.f32 s0, s2, s3
; HARD: vadd.f32 s0, s4, s0
  %0 = fmul float %a, %b
  %1 = fadd float %e, %0
  %2 = fmul float %c, %d
  %3 = fadd float %1, %2
  ret float %3
}

attributes #0 = { minsize nounwind optsize }
