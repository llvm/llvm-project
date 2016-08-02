; RUN: llc -mattr=+fp16 < %s | FileCheck %s --check-prefix=CHECK

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv7a--none-eabi"

; CHECK-LABEL: test_vec3:
; CHECK: vcvtb.f32.f16
; CHECK: vcvt.f32.s32
; CHECK: vadd.f32
; CHECK-NEXT: vcvtb.f16.f32 [[SREG:s[0-9]+]], {{.*}}
; CHECK-NEXT: vmov [[RREG1:r[0-9]+]], [[SREG]]
; CHECK-NEXT: uxth [[RREG2:r[0-9]+]], [[RREG1]]
; CHECK-NEXT: pkhbt [[RREG3:r[0-9]+]], [[RREG1]], [[RREG1]], lsl #16
; CHECK-DAG: strh [[RREG1]], [r0, #4]
; CHECK-DAG: vmov [[DREG:d[0-9]+]], [[RREG3]], [[RREG2]]
; CHECK-DAG: vst1.32 {[[DREG]][0]}, [r0:32]
; CHECK-NEXT: bx lr
define void @test_vec3(<3 x half>* %arr, i32 %i) #0 {
  %H = sitofp i32 %i to half
  %S = fadd half %H, 0xH4A00
  %1 = insertelement <3 x half> undef, half %S, i32 0
  %2 = insertelement <3 x half> %1, half %S, i32 1
  %3 = insertelement <3 x half> %2, half %S, i32 2
  store <3 x half> %3, <3 x half>* %arr, align 8
  ret void
}

attributes #0 = { nounwind }
