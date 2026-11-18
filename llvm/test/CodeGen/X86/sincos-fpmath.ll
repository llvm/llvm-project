; RUN: opt -mtriple=x86_64-unknown-linux-gnu -vector-library=LIBMVEC -passes=replace-with-veclib -S < %s | FileCheck %s

declare { <4 x float>, <4 x float> } @llvm.sincos.v4f32(<4 x float>)
declare { <2 x double>, <2 x double> } @llvm.sincos.v2f64(<2 x double>)
declare { <3 x float>, <3 x float> } @llvm.sincos.v3f32(<3 x float>)
declare void @use_sincos_struct({ <4 x float>, <4 x float> })

; v4f32 sincos -> _ZGVbN4v_sinf / _ZGVbN4v_cosf, both carrying !fpmath !0.
define void @sincos_fpmath_v4f32(<4 x float> %x, ptr noalias %sin_out, ptr noalias %cos_out) {
; CHECK-LABEL: @sincos_fpmath_v4f32(
; CHECK:         call <4 x float> @_ZGVbN4v_sinf(<4 x float> %x), !fpmath !0
; CHECK:         call <4 x float> @_ZGVbN4v_cosf(<4 x float> %x), !fpmath !0
;
  %r = call { <4 x float>, <4 x float> } @llvm.sincos.v4f32(<4 x float> %x), !fpmath !0
  %s = extractvalue { <4 x float>, <4 x float> } %r, 0
  %c = extractvalue { <4 x float>, <4 x float> } %r, 1
  store <4 x float> %s, ptr %sin_out, align 16
  store <4 x float> %c, ptr %cos_out, align 16
  ret void
}

; v2f64 sincos -> _ZGVbN2v_sin / _ZGVbN2v_cos, both carrying !fpmath !1.
define void @sincos_fpmath_v2f64(<2 x double> %x, ptr noalias %sin_out, ptr noalias %cos_out) {
; CHECK-LABEL: @sincos_fpmath_v2f64(
; CHECK:         call <2 x double> @_ZGVbN2v_sin(<2 x double> %x), !fpmath !1
; CHECK:         call <2 x double> @_ZGVbN2v_cos(<2 x double> %x), !fpmath !1
;
  %r = call { <2 x double>, <2 x double> } @llvm.sincos.v2f64(<2 x double> %x), !fpmath !1
  %s = extractvalue { <2 x double>, <2 x double> } %r, 0
  %c = extractvalue { <2 x double>, <2 x double> } %r, 1
  store <2 x double> %s, ptr %sin_out, align 16
  store <2 x double> %c, ptr %cos_out, align 16
  ret void
}

; When the original sincos has no fpmath metadata, the resulting vector sin
; and cos calls should also have none.
define void @sincos_no_fpmath_v4f32(<4 x float> %x, ptr noalias %sin_out, ptr noalias %cos_out) {
; CHECK-LABEL: @sincos_no_fpmath_v4f32(
; CHECK:         call <4 x float> @_ZGVbN4v_sinf(<4 x float> %x){{$}}
; CHECK-NOT:     !fpmath
; CHECK:         call <4 x float> @_ZGVbN4v_cosf(<4 x float> %x){{$}}
; CHECK-NOT:     !fpmath
;
  %r = call { <4 x float>, <4 x float> } @llvm.sincos.v4f32(<4 x float> %x)
  %s = extractvalue { <4 x float>, <4 x float> } %r, 0
  %c = extractvalue { <4 x float>, <4 x float> } %r, 1
  store <4 x float> %s, ptr %sin_out, align 16
  store <4 x float> %c, ptr %cos_out, align 16
  ret void
}

; A non-extractvalue user blocks the split, so llvm.sincos is left intact.
define void @sincos_non_extractvalue_user_v4f32(<4 x float> %x, ptr noalias %sin_out) {
; CHECK-LABEL: @sincos_non_extractvalue_user_v4f32(
; CHECK:         call { <4 x float>, <4 x float> } @llvm.sincos.v4f32(<4 x float> %x)
; CHECK-NOT:     _ZGV
; CHECK:         ret void
;
  %r = call { <4 x float>, <4 x float> } @llvm.sincos.v4f32(<4 x float> %x)
  call void @use_sincos_struct({ <4 x float>, <4 x float> } %r)
  %s = extractvalue { <4 x float>, <4 x float> } %r, 0
  store <4 x float> %s, ptr %sin_out, align 16
  ret void
}

; An odd vector width has no LIBMVEC sin/cos mapping, so llvm.sincos is left intact.
define void @sincos_no_veclib_mapping_v3f32(<3 x float> %x, ptr noalias %sin_out, ptr noalias %cos_out) {
; CHECK-LABEL: @sincos_no_veclib_mapping_v3f32(
; CHECK:         call { <3 x float>, <3 x float> } @llvm.sincos.v3f32(<3 x float> %x)
; CHECK-NOT:     _ZGV
; CHECK:         ret void
;
  %r = call { <3 x float>, <3 x float> } @llvm.sincos.v3f32(<3 x float> %x)
  %s = extractvalue { <3 x float>, <3 x float> } %r, 0
  %c = extractvalue { <3 x float>, <3 x float> } %r, 1
  store <3 x float> %s, ptr %sin_out, align 16
  store <3 x float> %c, ptr %cos_out, align 16
  ret void
}

; Verify the exact !fpmath metadata values are preserved.
; CHECK: !0 = !{float 2.500000e+00}
; CHECK: !1 = !{float 4.000000e+00}

!0 = !{float 2.5}
!1 = !{float 4.0}
