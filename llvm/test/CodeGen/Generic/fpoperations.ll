; RUN: llc < %s | FileCheck %s

; This test checks default lowering of the intrinsics operating floating point
; values. MSP430 is used as a target in this test because it does not have
; native FP support, so it won't get custom lowering for these intrinsics.
;
; REQUIRES: msp430-registered-target

target datalayout = "e-p:16:16:16-i8:8:8-i16:16:16-i32:16:32-n8:16"
target triple = "msp430---elf"


define float @roundeven_01(float %x) {
entry:
  %res = call float @llvm.roundeven.f32(float %x)
  ret float %res
}
; CHECK-LABEL: roundeven_01:
; CHECK: call #roundeven

declare float @llvm.roundeven.f32(float %x)
