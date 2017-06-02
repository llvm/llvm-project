; REQUIRES: asserts
; RUN: llc < %s -mtriple=armv8r-eabi -mcpu=cortex-a57 -enable-misched -verify-misched -debug-only=machine-scheduler -o - 2>&1 > /dev/null | FileCheck %s
; Check latencies of vmul/vfma accumulate chains.

define float @Test1(float %f1, float %f2, float %f3, float %f4, float %f5, float %f6) {
; CHECK:       ********** MI Scheduling **********
; CHECK:       Test1:BB#0

; CHECK:       VMULS
; > VMULS common latency = 5
; CHECK:       Latency            : 5
; CHECK:       Successors:
; CHECK:       data
; > VMULS read-advanced latency to VMLAS = 0
; CHECK-SAME:  Latency=0

; CHECK:       VMLAS
; > VMLAS common latency = 9
; CHECK:       Latency            : 9
; CHECK:       Successors:
; CHECK:       data
; > VMLAS read-advanced latency to the next VMLAS = 4
; CHECK-SAME:  Latency=4

; CHECK:       VMLAS
; CHECK:       Latency            : 9
; CHECK:       Successors:
; CHECK:       data
; > VMLAS not-optimized latency to VMOVRS = 9
; CHECK-SAME:  Latency=9

; f1 * f2 + f3 * f4 + f5 * f6  ==>  VMULS, VMLAS, VMLAS
  %mul1 = fmul float %f1, %f2
  %mul2 = fmul float %f3, %f4
  %mul3 = fmul float %f5, %f6
  %add1 = fadd float %mul1, %mul2
  %add2 = fadd float %add1, %mul3
  ret float %add2
}

; ASIMD form
define <2 x float> @Test2(<2 x float> %f1, <2 x float> %f2, <2 x float> %f3, <2 x float> %f4, <2 x float> %f5, <2 x float> %f6) {
; CHECK:       ********** MI Scheduling **********
; CHECK:       Test2:BB#0

; CHECK:       VMULfd
; > VMULfd common latency = 5
; CHECK:       Latency            : 5
; CHECK:       Successors:
; CHECK:       data
; VMULfd read-advanced latency to VMLAfd = 0
; CHECK-SAME:  Latency=0

; CHECK:       VMLAfd
; > VMLAfd common latency = 9
; CHECK:       Latency            : 9
; CHECK:       Successors:
; CHECK:       data
; > VMLAfd read-advanced latency to the next VMLAfd = 4
; CHECK-SAME:  Latency=4

; CHECK:       VMLAfd
; CHECK:       Latency            : 9
; CHECK:       Successors:
; CHECK:       data
; > VMLAfd not-optimized latency to VMOVRRD = 9
; CHECK-SAME:  Latency=9

; f1 * f2 + f3 * f4 + f5 * f6  ==>  VMULS, VMLAS, VMLAS
  %mul1 = fmul <2 x float> %f1, %f2
  %mul2 = fmul <2 x float> %f3, %f4
  %mul3 = fmul <2 x float> %f5, %f6
  %add1 = fadd <2 x float> %mul1, %mul2
  %add2 = fadd <2 x float> %add1, %mul3
  ret <2 x float> %add2
}

