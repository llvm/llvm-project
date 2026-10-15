; The "float-abi" module flag selects the floating-point calling convention
; used during LTO codegen, the same way the -float-abi codegen option does.

; REQUIRES: arm-registered-target

; RUN: split-file %s %t

; Hard float ABI: the argument and result stay in a VFP register (s0).
; RUN: llvm-as %t/hard.ll -o %t/hard.o
; RUN: llvm-lto2 run -r %t/hard.o,f,px %t/hard.o -filetype=asm -o %t/hard.s
; RUN: FileCheck %s --check-prefix=HARD < %t/hard.s.0

; Soft float ABI: the argument and result are passed in a GPR, so the value is
; moved between r0 and s0.
; RUN: llvm-as %t/soft.ll -o %t/soft.o
; RUN: llvm-lto2 run -r %t/soft.o,f,px %t/soft.o -filetype=asm -o %t/soft.s
; RUN: FileCheck %s --check-prefix=SOFT < %t/soft.s.0

;--- hard.ll
target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv7-none-eabi"

; HARD-LABEL: f:
; HARD-NEXT: .fnstart
; HARD-NEXT: vadd.f32 s0, s0, s0
define float @f(float %x) {
  %r = fadd float %x, %x
  ret float %r
}
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"float-abi", !"hard"}

;--- soft.ll
target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv7-none-eabi"

; SOFT-LABEL: f:
; SOFT-NEXT: .fnstart
; SOFT-NEXT: vmov s0, r0
define float @f(float %x) {
  %r = fadd float %x, %x
  ret float %r
}
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"float-abi", !"soft"}
