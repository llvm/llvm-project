; RUN: llc < %s -march=nvptx -mcpu=sm_10 | FileCheck %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_10 | FileCheck %s
; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s

define ptx_device float @test_fabsf(float %f) {
; CHECK: abs.f32 %f0, %f0;
; CHECK: ret;
	%x = call float @llvm.fabs.f32(float %f)
	ret float %x
}

define ptx_device double @test_fabs(double %d) {
; CHECK: abs.f64 %fl0, %fl0;
; CHECK: ret;
	%x = call double @llvm.fabs.f64(double %d)
	ret double %x
}

declare float @llvm.fabs.f32(float)
declare double @llvm.fabs.f64(double)
