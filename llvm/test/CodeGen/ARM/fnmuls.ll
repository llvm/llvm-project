; RUN: llc < %s -march=arm -mattr=+vfp2 | FileCheck %s
; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s
; RUN: llc < %s -march=arm -mcpu=cortex-a8 | FileCheck %s
; RUN: llc < %s -march=arm -mcpu=cortex-a9 | FileCheck %s

define arm_aapcs_vfpcc float @test1(float %a, float %b) nounwind {
; CHECK: vnmul.f32 s0, s0, s1 
entry:
	%0 = fmul float %a, %b
        %1 = fsub float -0.0, %0
	ret float %1
}

define arm_aapcs_vfpcc float @test2(float %a, float %b) nounwind {
; CHECK: vnmul.f32 s0, s0, s1 
entry:
	%0 = fmul float %a, %b
        %1 = fmul float -1.0, %0
	ret float %1
}

