; RUN: llc -mattr=+fp16 < %s | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv7a--none-eabi"

; CHECK-LABEL: test_vec3:
; CHECK-DAG: vmov.f32 [[SREG1:s[0-9]+]], #1.200000e+01
; CHECK-DAG: vcvt.f32.s32 [[SREG2:s[0-9]+]],
; CHECK-DAG: vcvtb.f16.f32 [[SREG3:s[0-9]+]], [[SREG2]]
; CHECK-DAG: vcvtb.f32.f16 [[SREG4:s[0-9]+]], [[SREG3]]
; CHECK: vadd.f32 [[SREG5:s[0-9]+]], [[SREG4]], [[SREG1]]
; CHECK-NEXT: vcvtb.f16.f32 [[SREG6:s[0-9]+]], [[SREG5]]
; CHECK-NEXT: vmov [[RREG1:r[0-9]+]], [[SREG6]]
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

; CHECK-LABEL: test_bitcast:
; CHECK: vcvtb.f16.f32
; CHECK: vcvtb.f16.f32
; CHECK: vcvtb.f16.f32
; CHECK: pkhbt
; CHECK: uxth
define void @test_bitcast(<3 x half> %inp, <3 x i16>* %arr) #0 {
  %bc = bitcast <3 x half> %inp to <3 x i16>
  store <3 x i16> %bc, <3 x i16>* %arr, align 8
  ret void
}

attributes #0 = { nounwind }
