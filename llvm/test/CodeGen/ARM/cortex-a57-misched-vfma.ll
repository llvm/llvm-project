; REQUIRES: asserts
; RUN: llc < %s -mtriple=armv8r-eabi -mcpu=cortex-a57 -enable-misched -verify-misched -debug-only=machine-scheduler -o - 2>&1 > /dev/null | FileCheck %s
; Check latencies of vmul/vfma accumulate chains. Without the contract flag the
; multiply/add pair lowers to the unfused VMLA/VMLS; with it, to the fused
; VFMA/VFMS/VFNMA/VFNMS.

define arm_aapcs_vfpcc float @Test1(float %f1, float %f2, float %f3, float %f4, float %f5, float %f6) {
; CHECK:       Current Schedule Region
; CHECK:       Test1:%bb.0

; CHECK:       VMULS
; > VMULS common latency = 5
; CHECK:       Latency            : 5
; CHECK:       Successors:
; CHECK:       Data
; > VMULS read-advanced latency to VMLAS = 0
; CHECK-SAME:  Latency=0

; CHECK:       VMLAS
; > VMLAS common latency = 9
; CHECK:       Latency            : 9
; CHECK:       Successors:
; CHECK:       Data
; > VMLAS read-advanced latency to the next VMLAS = 4
; CHECK-SAME:  Latency=4

; CHECK:       VMLAS
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

define arm_aapcs_vfpcc float @Test1_contract(float %f1, float %f2, float %f3, float %f4, float %f5, float %f6) {
; CHECK:       Current Schedule Region
; CHECK:       Test1_contract:%bb.0

; CHECK:       VMULS
; CHECK:       Latency            : 5
; CHECK:       Successors:
; CHECK:       Data
; CHECK-SAME:  Latency=0

; CHECK:       VFMAS
; CHECK:       Latency            : 9

  %mul1 = fmul contract float %f1, %f2
  %mul2 = fmul contract float %f3, %f4
  %mul3 = fmul contract float %f5, %f6
  %add1 = fadd contract float %mul1, %mul2
  %add2 = fadd contract float %add1, %mul3
  ret float %add2
}

; ASIMD form
define arm_aapcs_vfpcc <2 x float> @Test2(<2 x float> %f1, <2 x float> %f2, <2 x float> %f3, <2 x float> %f4, <2 x float> %f5, <2 x float> %f6) {
; CHECK:       Current Schedule Region
; CHECK:       Test2:%bb.0

; CHECK:       VMULfd
; > VMULfd common latency = 5
; CHECK:       Latency            : 5
; CHECK:       Successors:
; CHECK:       Data
; VMULfd read-advanced latency to VMLAfd = 0
; CHECK-SAME:  Latency=0

; CHECK:       VMLAfd
; > VMLAfd common latency = 9
; CHECK:       Latency            : 9
; CHECK:       Successors:
; CHECK:       Data
; > VMLAfd read-advanced latency to the next VMLAfd = 4
; CHECK-SAME:  Latency=4

; CHECK:       VMLAfd
; CHECK:       Latency            : 9
; CHECK:       Successors:
; CHECK:       Data
; > VMLAfd not-optimized latency to VMOVRRD = 9
; CHECK-SAME:  Latency=9

; f1 * f2 + f3 * f4 + f5 * f6  ==>  VMULfd, VMLAfd, VMLAfd
  %mul1 = fmul <2 x float> %f1, %f2
  %mul2 = fmul <2 x float> %f3, %f4
  %mul3 = fmul <2 x float> %f5, %f6
  %add1 = fadd <2 x float> %mul1, %mul2
  %add2 = fadd <2 x float> %add1, %mul3
  ret <2 x float> %add2
}

; ASIMD form
define arm_aapcs_vfpcc <2 x float> @Test2_contract(<2 x float> %f1, <2 x float> %f2, <2 x float> %f3, <2 x float> %f4, <2 x float> %f5, <2 x float> %f6) {
; CHECK:       Current Schedule Region
; CHECK:       Test2_contract:%bb.0

; CHECK:       VMULfd
; CHECK:       Latency            : 5
; CHECK:       Successors:
; CHECK:       Data
; CHECK-SAME:  Latency=0

; CHECK:       VFMAfd
; CHECK:       Latency            : 9

  %mul1 = fmul contract <2 x float> %f1, %f2
  %mul2 = fmul contract <2 x float> %f3, %f4
  %mul3 = fmul contract <2 x float> %f5, %f6
  %add1 = fadd contract <2 x float> %mul1, %mul2
  %add2 = fadd contract <2 x float> %add1, %mul3
  ret <2 x float> %add2
}

define arm_aapcs_vfpcc float @Test3(float %f1, float %f2, float %f3, float %f4, float %f5, float %f6) {
; CHECK:       Current Schedule Region
; CHECK:       Test3:%bb.0

; CHECK:       VMULS
; > VMULS common latency = 5
; CHECK:       Latency            : 5
; CHECK:       Successors:
; CHECK:       Data
; > VMULS read-advanced latency to VMLSS = 0
; CHECK-SAME:  Latency=0

; CHECK:       VMLSS
; CHECK:       Latency            : 9
; CHECK:       Successors:
; CHECK:       Data
; > VMLSS read-advanced latency to the next VMLSS = 4
; CHECK-SAME:  Latency=4

; CHECK:       VMLSS
; CHECK:       Latency            : 9
; CHECK:       Successors:
; CHECK:       Data
; > VMLSS not-optimized latency to VMOVRS = 9
; CHECK-SAME:  Latency=9

; f1 * f2 - f3 * f4 - f5 * f6  ==>  VMULS, VMLSS, VMLSS
  %mul1 = fmul float %f1, %f2
  %mul2 = fmul float %f3, %f4
  %mul3 = fmul float %f5, %f6
  %sub1 = fsub float %mul1, %mul2
  %sub2 = fsub float %sub1, %mul3
  ret float %sub2
}

define arm_aapcs_vfpcc float @Test3_contract(float %f1, float %f2, float %f3, float %f4, float %f5, float %f6) {
; CHECK:       Current Schedule Region
; CHECK:       Test3_contract:%bb.0

; CHECK:       VMULS
; CHECK:       Latency            : 5
; CHECK:       Successors:
; CHECK:       Data
; CHECK-SAME:  Latency=0

; CHECK:       VFNMSS
; CHECK:       Latency            : 9

  %mul1 = fmul contract float %f1, %f2
  %mul2 = fmul contract float %f3, %f4
  %mul3 = fmul contract float %f5, %f6
  %sub1 = fsub contract float %mul1, %mul2
  %sub2 = fsub contract float %sub1, %mul3
  ret float %sub2
}

; ASIMD form
define arm_aapcs_vfpcc <2 x float> @Test4(<2 x float> %f1, <2 x float> %f2, <2 x float> %f3, <2 x float> %f4, <2 x float> %f5, <2 x float> %f6) {
; CHECK:       Current Schedule Region
; CHECK:       Test4:%bb.0

; CHECK:       VMULfd
; > VMULfd common latency = 5
; CHECK:       Latency            : 5
; CHECK:       Successors:
; CHECK:       Data
; VMULfd read-advanced latency to VMLSfd = 0
; CHECK-SAME:  Latency=0

; CHECK:       VMLSfd
; > VMLSfd common latency = 9
; CHECK:       Latency            : 9
; CHECK:       Successors:
; CHECK:       Data
; > VMLSfd read-advanced latency to the next VMLSfd = 4
; CHECK-SAME:  Latency=4

; CHECK:       VMLSfd
; CHECK:       Latency            : 9
; CHECK:       Successors:
; CHECK:       Data
; > VMLSfd not-optimized latency to VMOVRRD = 9
; CHECK-SAME:  Latency=9

; f1 * f2 - f3 * f4 - f5 * f6  ==>  VMULfd, VMLSfd, VMLSfd
  %mul1 = fmul <2 x float> %f1, %f2
  %mul2 = fmul <2 x float> %f3, %f4
  %mul3 = fmul <2 x float> %f5, %f6
  %sub1 = fsub <2 x float> %mul1, %mul2
  %sub2 = fsub <2 x float> %sub1, %mul3
  ret <2 x float> %sub2
}

; ASIMD form
define arm_aapcs_vfpcc <2 x float> @Test4_contract(<2 x float> %f1, <2 x float> %f2, <2 x float> %f3, <2 x float> %f4, <2 x float> %f5, <2 x float> %f6) {
; CHECK:       Current Schedule Region
; CHECK:       Test4_contract:%bb.0

; CHECK:       VMULfd
; CHECK:       Latency            : 5
; CHECK:       Successors:
; CHECK:       Data
; CHECK-SAME:  Latency=0

; CHECK:       VFMSfd
; CHECK:       Latency            : 9

  %mul1 = fmul contract <2 x float> %f1, %f2
  %mul2 = fmul contract <2 x float> %f3, %f4
  %mul3 = fmul contract <2 x float> %f5, %f6
  %sub1 = fsub contract <2 x float> %mul1, %mul2
  %sub2 = fsub contract <2 x float> %sub1, %mul3
  ret <2 x float> %sub2
}

define arm_aapcs_vfpcc float @Test5(float %f1, float %f2, float %f3) {
; CHECK:       Current Schedule Region
; CHECK:       Test5:%bb.0

; CHECK:       VNMLS
; CHECK:       Latency            : 9
; CHECK:       Successors:
; CHECK:       Data
; > VMLAS not-optimized latency to VMOVRS = 9
; CHECK-SAME:  Latency=9

; f1 * f2 - f3  ==>  VNMLS
  %mul = fmul float %f1, %f2
  %sub = fsub float %mul, %f3
  ret float %sub
}

define arm_aapcs_vfpcc float @Test5_contract(float %f1, float %f2, float %f3) {
; CHECK:       Current Schedule Region
; CHECK:       Test5_contract:%bb.0

; CHECK:       VFNMS
; CHECK:       Latency            : 9

; f1 * f2 - f3  ==>  VFNMS
  %mul = fmul contract float %f1, %f2
  %sub = fsub contract float %mul, %f3
  ret float %sub
}

define arm_aapcs_vfpcc float @Test6(float %f1, float %f2, float %f3) {
; CHECK:       Current Schedule Region
; CHECK:       Test6:%bb.0

; CHECK:       VNMLA
; CHECK:       Latency            : 9
; CHECK:       Successors:
; CHECK:       Data
; > VMLAS not-optimized latency to VMOVRS = 9
; CHECK-SAME:  Latency=9

; -(f1 * f2) - f3  ==>  VNMLA
  %mul = fmul float %f1, %f2
  %sub1 = fsub float -0.0, %mul
  %sub2 = fsub float %sub1, %f2
  ret float %sub2
}

define arm_aapcs_vfpcc float @Test6_contract(float %f1, float %f2, float %f3) {
; CHECK:       Current Schedule Region
; CHECK:       Test6_contract:%bb.0

; CHECK:       VFNMA
; CHECK:       Latency            : 9

; -(f1 * f2) - f3  ==>  VFNMA
  %mul = fmul contract float %f1, %f2
  %sub1 = fsub contract float -0.0, %mul
  %sub2 = fsub contract float %sub1, %f2
  ret float %sub2
}
