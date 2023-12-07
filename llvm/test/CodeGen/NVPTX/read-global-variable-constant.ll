; RUN: llc < %s -march=nvptx64 -mcpu=sm_35 -verify-machineinstrs | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_35 | %ptxas-verify %}

; Check load from constant global variables.  These loads should be
; ld.global.nc (aka ldg).

@gv_float = external constant float
@gv_float2 = external constant <2 x float>
@gv_float4 = external constant <4 x float>

; CHECK-LABEL: test_gv_float()
define float @test_gv_float() {
; CHECK: ld.global.nc.f32
  %v = load float, ptr @gv_float
  ret float %v
}

; CHECK-LABEL: test_gv_float2()
define <2 x float> @test_gv_float2() {
; CHECK: ld.global.nc.v2.f32
  %v = load <2 x float>, ptr @gv_float2
  ret <2 x float> %v
}

; CHECK-LABEL: test_gv_float4()
define <4 x float> @test_gv_float4() {
; CHECK: ld.global.nc.v4.f32
  %v = load <4 x float>, ptr @gv_float4
  ret <4 x float> %v
}
