; RUN: opt -S -march=r600 -mcpu=cayman -basicaa -slp-vectorizer -dce < %s | FileCheck %s
; XFAIL: *
; 
; FIXME: If this test expects to be vectorized, the TTI must indicate that the target
;        has vector registers of the expected width.
;        Currently, it says there are 8 vector registers that are 32-bits wide.

target datalayout = "e-p:32:32:32-p3:16:16:16-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-v2048:2048:2048-n32:64"


; Simple 3-pair chain with loads and stores
define amdgpu_kernel void @test1_as_3_3_3(double addrspace(3)* %a, double addrspace(3)* %b, double addrspace(3)* %c) {
; CHECK-LABEL: @test1_as_3_3_3(
; CHECK: load <2 x double>, <2 x double> addrspace(3)*
; CHECK: load <2 x double>, <2 x double> addrspace(3)*
; CHECK: store <2 x double> %{{.*}}, <2 x double> addrspace(3)* %
; CHECK: ret
  %i0 = load double, double addrspace(3)* %a, align 8
  %i1 = load double, double addrspace(3)* %b, align 8
  %mul = fmul double %i0, %i1
  %arrayidx3 = getelementptr inbounds double, double addrspace(3)* %a, i64 1
  %i3 = load double, double addrspace(3)* %arrayidx3, align 8
  %arrayidx4 = getelementptr inbounds double, double addrspace(3)* %b, i64 1
  %i4 = load double, double addrspace(3)* %arrayidx4, align 8
  %mul5 = fmul double %i3, %i4
  store double %mul, double addrspace(3)* %c, align 8
  %arrayidx5 = getelementptr inbounds double, double addrspace(3)* %c, i64 1
  store double %mul5, double addrspace(3)* %arrayidx5, align 8
  ret void
}

define amdgpu_kernel void @test1_as_3_0_0(double addrspace(3)* %a, double* %b, double* %c) {
; CHECK-LABEL: @test1_as_3_0_0(
; CHECK: load <2 x double>, <2 x double> addrspace(3)*
; CHECK: load <2 x double>, <2 x double>*
; CHECK: store <2 x double> %{{.*}}, <2 x double>* %
; CHECK: ret
  %i0 = load double, double addrspace(3)* %a, align 8
  %i1 = load double, double* %b, align 8
  %mul = fmul double %i0, %i1
  %arrayidx3 = getelementptr inbounds double, double addrspace(3)* %a, i64 1
  %i3 = load double, double addrspace(3)* %arrayidx3, align 8
  %arrayidx4 = getelementptr inbounds double, double* %b, i64 1
  %i4 = load double, double* %arrayidx4, align 8
  %mul5 = fmul double %i3, %i4
  store double %mul, double* %c, align 8
  %arrayidx5 = getelementptr inbounds double, double* %c, i64 1
  store double %mul5, double* %arrayidx5, align 8
  ret void
}

define amdgpu_kernel void @test1_as_0_0_3(double* %a, double* %b, double addrspace(3)* %c) {
; CHECK-LABEL: @test1_as_0_0_3(
; CHECK: load <2 x double>, <2 x double>*
; CHECK: load <2 x double>, <2 x double>*
; CHECK: store <2 x double> %{{.*}}, <2 x double> addrspace(3)* %
; CHECK: ret
  %i0 = load double, double* %a, align 8
  %i1 = load double, double* %b, align 8
  %mul = fmul double %i0, %i1
  %arrayidx3 = getelementptr inbounds double, double* %a, i64 1
  %i3 = load double, double* %arrayidx3, align 8
  %arrayidx4 = getelementptr inbounds double, double* %b, i64 1
  %i4 = load double, double* %arrayidx4, align 8
  %mul5 = fmul double %i3, %i4
  store double %mul, double addrspace(3)* %c, align 8
  %arrayidx5 = getelementptr inbounds double, double addrspace(3)* %c, i64 1
  store double %mul5, double addrspace(3)* %arrayidx5, align 8
  ret void
}
