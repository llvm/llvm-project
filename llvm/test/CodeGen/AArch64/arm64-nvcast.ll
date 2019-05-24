; RUN: llc < %s -mtriple=arm64-apple-ios | FileCheck %s

; CHECK-LABEL: _test:
; CHECK-DAG:  fmov.2d v0, #2.00000000
; CHECK-DAG: and [[MASK_IDX:x[0-9]+]], x1, #0x3
; CHECK-DAG:  mov  x9, sp
; CHECK-DAG:  str  q0, [sp], #16
; CHECK-DAG:  bfi [[PTR:x[0-9]+]], [[MASK_IDX]], #2, #2
; CHECK:  ldr s0, {{\[}}[[PTR]]{{\]}}
; CHECK:  str  s0, [x0]

define void @test(float * %p1, i32 %v1) {
entry:
  %v2 = extractelement <3 x float> <float 0.000000e+00, float 2.000000e+00, float 0.000000e+00>, i32 %v1
  store float %v2, float* %p1, align 4
  ret void
}

; CHECK-LABEL: _test2
; CHECK: movi.16b  v0, #63
; CHECK-DAG: and [[MASK_IDX:x[0-9]+]], x1, #0x3
; CHECK-DAG: str  q0, [sp], #16
; CHECK-DAG: mov  x9, sp
; CHECK-DAG:  bfi [[PTR:x[0-9]+]], [[MASK_IDX]], #2, #2
; CHECK: ldr s0, {{\[}}[[PTR]]{{\]}}
; CHECK: str  s0, [x0]

define void @test2(float * %p1, i32 %v1) {
entry:
  %v2 = extractelement <3 x float> <float 0.7470588088035583, float 0.7470588088035583, float 0.7470588088035583>, i32 %v1
  store float %v2, float* %p1, align 4
  ret void
}


%"st1" = type { %"subst1", %"subst1", %"subst1" }
%"subst1" = type { %float4 }
%float4 = type { float, float, float, float }

@_gv = external unnamed_addr global %"st1", align 8

define internal void @nvcast_f32_v8i8() {
; CHECK-LABEL: _nvcast_f32_v8i8
; CHECK: movi.8b v[[REG:[0-9]+]], #254
; CHECK: str d[[REG]]
entry:
  store <2 x float> <float 0xC7DFDFDFC0000000, float 0xC7DFDFDFC0000000>, <2 x float>* bitcast (%"st1"* @_gv to <2 x float>*), align 8
  ret void
}

%struct.Vector3 = type { float, float, float }

define void @nvcast_v2f32_v1f64(%struct.Vector3*) {
; CHECK-LABEL: _nvcast_v2f32_v1f64
; CHECK: fmov.2s v[[REG:[0-9]+]], #1.00000000
; CHECK: str d[[REG]], [x0]
entry:
  %a13 = bitcast %struct.Vector3* %0 to <1 x double>*
  store <1 x double> <double 0x3F8000003F800000>, <1 x double>* %a13, align 8
  ret void
}
