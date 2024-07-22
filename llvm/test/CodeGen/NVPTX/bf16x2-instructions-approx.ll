; RUN: llc < %s -march=nvptx64 -mcpu=sm_80 -mattr=+ptx71 --enable-unsafe-fp-math | FileCheck --check-prefixes=CHECK %s
; RUN: %if ptxas-11.8 %{ llc < %s -march=nvptx64 -mcpu=sm_80 -mattr=+ptx71 --enable-unsafe-fp-math | %ptxas-verify -arch=sm_80 %}

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

declare <2 x bfloat> @llvm.sin.f16(<2 x bfloat> %a) #0
declare <2 x bfloat> @llvm.cos.f16(<2 x bfloat> %a) #0

; CHECK-LABEL: test_sin(
; CHECK:      ld.param.b32    [[A:%r[0-9]+]], [test_sin_param_0];
; CHECK:      mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-DAG:  cvt.f32.bf16     [[AF0:%f[0-9]+]], [[A0]];
; CHECK-DAG:  cvt.f32.bf16     [[AF1:%f[0-9]+]], [[A1]];
; CHECK-DAG:  sin.approx.f32  [[RF0:%f[0-9]+]], [[AF0]];
; CHECK-DAG:  sin.approx.f32  [[RF1:%f[0-9]+]], [[AF1]];
; CHECK-DAG:  cvt.rn.bf16.f32  [[R0:%rs[0-9]+]], [[RF0]];
; CHECK-DAG:  cvt.rn.bf16.f32  [[R1:%rs[0-9]+]], [[RF1]];
; CHECK:      mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]}
; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;
define <2 x bfloat> @test_sin(<2 x bfloat> %a) #0 #1 {
  %r = call <2 x bfloat> @llvm.sin.f16(<2 x bfloat> %a)
  ret <2 x bfloat> %r
}

; CHECK-LABEL: test_cos(
; CHECK:      ld.param.b32    [[A:%r[0-9]+]], [test_cos_param_0];
; CHECK:      mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-DAG:  cvt.f32.bf16     [[AF0:%f[0-9]+]], [[A0]];
; CHECK-DAG:  cvt.f32.bf16     [[AF1:%f[0-9]+]], [[A1]];
; CHECK-DAG:  cos.approx.f32  [[RF0:%f[0-9]+]], [[AF0]];
; CHECK-DAG:  cos.approx.f32  [[RF1:%f[0-9]+]], [[AF1]];
; CHECK-DAG:  cvt.rn.bf16.f32  [[R0:%rs[0-9]+]], [[RF0]];
; CHECK-DAG:  cvt.rn.bf16.f32  [[R1:%rs[0-9]+]], [[RF1]];
; CHECK:      mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]}
; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;
define <2 x bfloat> @test_cos(<2 x bfloat> %a) #0 #1 {
  %r = call <2 x bfloat> @llvm.cos.f16(<2 x bfloat> %a)
  ret <2 x bfloat> %r
}

