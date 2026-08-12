; The "float-abi" module flag selects the floating-point calling convention.
; RUN: split-file %s %t

; Hard float ABI module flag: FP argument returned in a VFP register (s0).
; RUN: llc -mtriple=armv7-none-eabi -mattr=+vfp3 < %t/hard.ll | FileCheck %s --check-prefix=HARD

; Soft float ABI module flag: FP argument returned in a GPR (r0).
; RUN: llc -mtriple=armv7-none-eabi -mattr=+vfp3 < %t/soft.ll | FileCheck %s --check-prefix=SOFT

; No module flag: the -float-abi command-line option still applies.
; RUN: llc -mtriple=armv7-none-eabi -mattr=+vfp3 -float-abi=hard < %t/none.ll | FileCheck %s --check-prefix=HARD
; RUN: llc -mtriple=armv7-none-eabi -mattr=+vfp3 -float-abi=soft < %t/none.ll | FileCheck %s --check-prefix=SOFT

; The triple default applies with no module flag.
; RUN: llc -mtriple=armv7-none-eabi -mattr=+vfp3 < %t/none.ll | FileCheck %s --check-prefix=SOFT

; An explicit module flag takes precedence over a conflicting -float-abi option.
; RUN: llc -mtriple=armv7-none-eabi -mattr=+vfp3 -float-abi=soft < %t/hard.ll | FileCheck %s --check-prefix=HARD
; RUN: llc -mtriple=armv7-none-eabi -mattr=+vfp3 -float-abi=hard < %t/soft.ll | FileCheck %s --check-prefix=SOFT

;--- hard.ll
define float @f(float %x) {
  %r = fadd float %x, %x
  ret float %r
}
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"float-abi", !"hard"}
; HARD: vadd.f32 s0,

;--- soft.ll
define float @f(float %x) {
  %r = fadd float %x, %x
  ret float %r
}
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"float-abi", !"soft"}
; SOFT: vmov {{s[0-9]+}}, r0

;--- none.ll
define float @f(float %x) {
  %r = fadd float %x, %x
  ret float %r
}
