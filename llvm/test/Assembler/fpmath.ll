; RUN: llvm-as < %s | llvm-dis | FileCheck %s

define { float, float } @use_fpmath_struct(float %x) {
; CHECK-LABEL: @use_fpmath_struct(
; CHECK: %ret = call { float, float } @llvm.sincos.f32(float %x), !fpmath !0
  %ret = call { float, float } @llvm.sincos.f32(float %x), !fpmath !0
  ret { float, float } %ret
}

define { <2 x float>, <2 x float> } @use_fpmath_struct_vec(<2 x float> %x) {
; CHECK-LABEL: @use_fpmath_struct_vec(
; CHECK: %ret = call { <2 x float>, <2 x float> } @llvm.sincos.v2f32(<2 x float> %x), !fpmath !0
  %ret = call { <2 x float>, <2 x float> } @llvm.sincos.v2f32(<2 x float> %x), !fpmath !0
  ret { <2 x float>, <2 x float> } %ret
}

!0 = !{float 4.0}

; CHECK: !0 = !{float 4.000000e+00}
