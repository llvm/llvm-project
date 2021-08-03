; RUN: llc < %s -mtriple=arm64-apple-ios | FileCheck %s

define void @test(float * %p1, i32 %v1) {
; CHECK-LABEL: test:
; CHECK:       ; %bb.0: ; %entry
; CHECK-NEXT:    sub sp, sp, #16
; CHECK-NEXT:    .cfi_def_cfa_offset 16
; CHECK-NEXT:    ; kill: def $w1 killed $w1 def $x1
; CHECK-NEXT:    fmov.2d v0, #2.00000000
; CHECK-NEXT:    and x8, x1, #0x3
; CHECK-NEXT:    mov x9, sp
; CHECK-NEXT:    str q0, [sp]
; CHECK-NEXT:    bfi x9, x8, #2, #2
; CHECK-NEXT:    ldr s0, [x9]
; CHECK-NEXT:    str s0, [x0]
; CHECK-NEXT:    add sp, sp, #16
; CHECK-NEXT:    ret
entry:
  %v2 = extractelement <3 x float> <float 0.000000e+00, float 2.000000e+00, float 0.000000e+00>, i32 %v1
  store float %v2, float* %p1, align 4
  ret void
}

define void @test2(float * %p1, i32 %v1) {
; CHECK-LABEL: test2:
; CHECK:       ; %bb.0: ; %entry
; CHECK-NEXT:    sub sp, sp, #16
; CHECK-NEXT:    .cfi_def_cfa_offset 16
; CHECK-NEXT:    ; kill: def $w1 killed $w1 def $x1
; CHECK-NEXT:    movi.16b v0, #63
; CHECK-NEXT:    and x8, x1, #0x3
; CHECK-NEXT:    mov x9, sp
; CHECK-NEXT:    str q0, [sp]
; CHECK-NEXT:    bfi x9, x8, #2, #2
; CHECK-NEXT:    ldr s0, [x9]
; CHECK-NEXT:    str s0, [x0]
; CHECK-NEXT:    add sp, sp, #16
; CHECK-NEXT:    ret
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
