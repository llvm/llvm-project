; RUN: llc -mtriple=aarch64-none-linux-gnu < %s | FileCheck %s

; C frexp/frexpf write the exponent through an int * parameter (32 bits),
; but the declared exponent type may be wider (i64) or narrower (i8/i16).
; The stack temporary for the libcall must match the libcall's parameter
; width, and the declared result must be materialized from that 32-bit
; value (no full-width reads from the narrow libcall write).

declare { float, i64 } @llvm.frexp.f32.i64(float)
declare { double, i64 } @llvm.frexp.f64.i64(double)
declare { float, i32 } @llvm.frexp.f32.i32(float)
declare { float, i8 } @llvm.frexp.f32.i8(float)
declare { <2 x float>, <2 x i64> } @llvm.frexp.v2f32.v2i64(<2 x float>)

; The 32-bit libcall output must be sign-extended when materializing i64.
define i64 @exp64_f32(float %x) {
; CHECK-LABEL: exp64_f32:
; CHECK: bl frexpf
; CHECK: ldrsw x{{[0-9]+}}, [sp, #[[OFF:[0-9]+]]]
  %r = call { float, i64 } @llvm.frexp.f32.i64(float %x)
  %e = extractvalue { float, i64 } %r, 1
  ret i64 %e
}

define i64 @exp64_f64(double %x) {
; CHECK-LABEL: exp64_f64:
; CHECK: bl frexp
; CHECK: ldrsw x{{[0-9]+}}, [sp, #
  %r = call { double, i64 } @llvm.frexp.f64.i64(double %x)
  %e = extractvalue { double, i64 } %r, 1
  ret i64 %e
}

; i32 exponent: the libcall parameter width matches the declared type.
; This must remain unchanged.
define i32 @exp32_f32(float %x) {
; CHECK-LABEL: exp32_f32:
; CHECK: bl frexpf
; CHECK: ldr w{{[0-9]+}}, [sp, #
  %r = call { float, i32 } @llvm.frexp.f32.i32(float %x)
  %e = extractvalue { float, i32 } %r, 1
  ret i32 %e
}

; Narrow declared exponent (i8): the libcall still writes C int, so the
; temporary must be at least as wide as C int and the result is truncated
; from it (matching constant folding).
define i8 @exp8_f32(float %x) {
; CHECK-LABEL: exp8_f32:
; CHECK: bl frexpf
; CHECK: ldr w{{[0-9]+}}, [sp, #
  %r = call { float, i8 } @llvm.frexp.f32.i8(float %x)
  %e = extractvalue { float, i8 } %r, 1
  ret i8 %e
}

; Vector form is scalarized per lane; each lane reads back 32-bit values.
define <2 x i64> @exp64_v2f32(<2 x float> %x) {
; CHECK-LABEL: exp64_v2f32:
; CHECK-COUNT-2: bl frexpf
; CHECK: ldpsw
  %r = call { <2 x float>, <2 x i64> } @llvm.frexp.v2f32.v2i64(<2 x float> %x)
  %e = extractvalue { <2 x float>, <2 x i64> } %r, 1
  ret <2 x i64> %e
}

@gm = global float 0.000000e+00
@ge = global i64 0

; The exponent result feeds a plain store whose width (i64) differs from
; the libcall's parameter width: the libcall must NOT write into the
; destination directly; the full-width store to the destination must remain.
define void @both(float %x) {
; CHECK-LABEL: both:
; CHECK: bl frexpf
; CHECK: ldrsw x{{[0-9]+}}, [sp, #
; CHECK: str {{x[0-9]+}}, [
  %r = call { float, i64 } @llvm.frexp.f32.i64(float %x)
  %m = extractvalue { float, i64 } %r, 0
  %e = extractvalue { float, i64 } %r, 1
  store float %m, ptr @gm
  store i64 %e, ptr @ge
  ret void
}

; MODF (C modf) writes the FP type itself through its output pointer; its
; codegen is unaffected by this change.
declare { float, float } @llvm.modf.f32(float)
@gmi = global float 0.000000e+00
@gmf = global float 0.000000e+00

define void @mf(float %x) {
; CHECK-LABEL: mf:
; CHECK: bl modff
  %r = call { float, float } @llvm.modf.f32(float %x)
  %i = extractvalue { float, float } %r, 0
  %f = extractvalue { float, float } %r, 1
  store float %i, ptr @gmi
  store float %f, ptr @gmf
  ret void
}
