; RUN: llc < %s -march=nvptx -mcpu=sm_35 | FileCheck %s

; Check load from constant global variables.  These loads should be
; ld.global.nc (aka ldg).

@gv_float = external constant float
@gv_float2 = external constant <2 x float>
@gv_float4 = external constant <4 x float>

; CHECK-LABEL: test_gv_float()
define float @test_gv_float() {
; CHECK: ld.global.nc.f32
  %v = load float, float* @gv_float
  ret float %v
}

; CHECK-LABEL: test_gv_float2()
define <2 x float> @test_gv_float2() {
; CHECK: ld.global.nc.v2.f32
  %v = load <2 x float>, <2 x float>* @gv_float2
  ret <2 x float> %v
}

; CHECK-LABEL: test_gv_float4()
define <4 x float> @test_gv_float4() {
; CHECK: ld.global.nc.v4.f32
  %v = load <4 x float>, <4 x float>* @gv_float4
  ret <4 x float> %v
}
