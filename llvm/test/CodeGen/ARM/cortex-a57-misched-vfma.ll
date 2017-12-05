; REQUIRES: asserts
; RUN: llc < %s -mtriple=armv8r-eabi -mcpu=cortex-a57 -enable-misched -verify-misched -debug-only=machine-scheduler -o - 2>&1 > /dev/null | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-DEFAULT
; RUN: llc < %s -mtriple=armv8r-eabi -mcpu=cortex-a57 -enable-misched -verify-misched -debug-only=machine-scheduler -o - 2>&1 > /dev/null -fp-contract=fast | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-FAST
; Check latencies of vmul/vfma accumulate chains.

define float @Test1(float %f1, float %f2, float %f3, float %f4, float %f5, float %f6) {
; CHECK:       ********** MI Scheduling **********
; CHECK:       Test1:%bb.0

; CHECK:       VMULS
; > VMULS common latency = 5
; CHECK:       Latency            : 5
; CHECK:       Successors:
; CHECK:       Data
; > VMULS read-advanced latency to VMLAS = 0
; CHECK-SAME:  Latency=0

; CHECK-DEFAULT: VMLAS
; CHECK-FAST:    VFMAS
; > VMLAS common latency = 9
; CHECK:       Latency            : 9
; CHECK:       Successors:
; CHECK:       Data
; > VMLAS read-advanced latency to the next VMLAS = 4
; CHECK-SAME:  Latency=4

; CHECK-DEFAULT: VMLAS
; CHECK-FAST:    VFMAS
; CHECK:       Latency            : 9
; CHECK:       Successors:
; CHECK:       Data
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
; CHECK:       Test2:%bb.0

; CHECK:       VMULfd
; > VMULfd common latency = 5
; CHECK:       Latency            : 5
; CHECK:       Successors:
; CHECK:       Data
; VMULfd read-advanced latency to VMLAfd = 0
; CHECK-SAME:  Latency=0

; CHECK-DEFAULT: VMLAfd
; CHECK-FAST:    VFMAfd
; > VMLAfd common latency = 9
; CHECK:       Latency            : 9
; CHECK:       Successors:
; CHECK:       Data
; > VMLAfd read-advanced latency to the next VMLAfd = 4
; CHECK-SAME:  Latency=4

; CHECK-DEFAULT: VMLAfd
; CHECK-FAST:    VFMAfd
; CHECK:       Latency            : 9
; CHECK:       Successors:
; CHECK:       Data
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

define float @Test3(float %f1, float %f2, float %f3, float %f4, float %f5, float %f6) {
; CHECK:       ********** MI Scheduling **********
; CHECK:       Test3:%bb.0

; CHECK:       VMULS
; > VMULS common latency = 5
; CHECK:       Latency            : 5
; CHECK:       Successors:
; CHECK:       Data
; > VMULS read-advanced latency to VMLSS = 0
; CHECK-SAME:  Latency=0

; CHECK-DEFAULT: VMLSS
; CHECK-FAST:    VFMSS
; > VMLSS common latency = 9
; CHECK:       Latency            : 9
; CHECK:       Successors:
; CHECK:       Data
; > VMLSS read-advanced latency to the next VMLSS = 4
; CHECK-SAME:  Latency=4

; CHECK-DEFAULT: VMLSS
; CHECK-FAST:    VFMSS
; CHECK:       Latency            : 9
; CHECK:       Successors:
; CHECK:       Data
; > VMLSS not-optimized latency to VMOVRS = 9
; CHECK-SAME:  Latency=9

; f1 * f2 + f3 * f4 + f5 * f6  ==>  VMULS, VMLSS, VMLSS
  %mul1 = fmul float %f1, %f2
  %mul2 = fmul float %f3, %f4
  %mul3 = fmul float %f5, %f6
  %sub1 = fsub float %mul1, %mul2
  %sub2 = fsub float %sub1, %mul3
  ret float %sub2
}

; ASIMD form
define <2 x float> @Test4(<2 x float> %f1, <2 x float> %f2, <2 x float> %f3, <2 x float> %f4, <2 x float> %f5, <2 x float> %f6) {
; CHECK:       ********** MI Scheduling **********
; CHECK:       Test4:%bb.0

; CHECK:       VMULfd
; > VMULfd common latency = 5
; CHECK:       Latency            : 5
; CHECK:       Successors:
; CHECK:       Data
; VMULfd read-advanced latency to VMLSfd = 0
; CHECK-SAME:  Latency=0

; CHECK-DEFAULT: VMLSfd
; CHECK-FAST:    VFMSfd
; > VMLSfd common latency = 9
; CHECK:       Latency            : 9
; CHECK:       Successors:
; CHECK:       Data
; > VMLSfd read-advanced latency to the next VMLSfd = 4
; CHECK-SAME:  Latency=4

; CHECK-DEFAULT: VMLSfd
; CHECK-FAST:    VFMSfd
; CHECK:       Latency            : 9
; CHECK:       Successors:
; CHECK:       Data
; > VMLSfd not-optimized latency to VMOVRRD = 9
; CHECK-SAME:  Latency=9

; f1 * f2 + f3 * f4 + f5 * f6  ==>  VMULS, VMLSS, VMLSS
  %mul1 = fmul <2 x float> %f1, %f2
  %mul2 = fmul <2 x float> %f3, %f4
  %mul3 = fmul <2 x float> %f5, %f6
  %sub1 = fsub <2 x float> %mul1, %mul2
  %sub2 = fsub <2 x float> %sub1, %mul3
  ret <2 x float> %sub2
}

define float @Test5(float %f1, float %f2, float %f3) {
; CHECK:       ********** MI Scheduling **********
; CHECK:       Test5:%bb.0

; CHECK-DEFAULT: VNMLS
; CHECK-FAST:    VFNMS
; CHECK:       Latency            : 9
; CHECK:       Successors:
; CHECK:       Data
; > VMLAS not-optimized latency to VMOVRS = 9
; CHECK-SAME:  Latency=9

; f1 * f2 - f3  ==>  VNMLS/VFNMS
  %mul = fmul float %f1, %f2
  %sub = fsub float %mul, %f3
  ret float %sub
}


define float @Test6(float %f1, float %f2, float %f3) {
; CHECK:       ********** MI Scheduling **********
; CHECK:       Test6:%bb.0

; CHECK-DEFAULT: VNMLA
; CHECK-FAST:    VFNMA
; CHECK:       Latency            : 9
; CHECK:       Successors:
; CHECK:       Data
; > VMLAS not-optimized latency to VMOVRS = 9
; CHECK-SAME:  Latency=9

; f1 * f2 - f3  ==>  VNMLA/VFNMA
  %mul = fmul float %f1, %f2
  %sub1 = fsub float -0.0, %mul
  %sub2 = fsub float %sub1, %f2
  ret float %sub2
}
