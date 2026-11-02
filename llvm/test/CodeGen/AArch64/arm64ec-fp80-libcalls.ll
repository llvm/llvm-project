; RUN: llc < %s -mtriple=arm64ec-pc-windows-msvc -O0 -o - | FileCheck %s

; On ARM64EC, x86_fp80 is not a legal type: it is softened to i128 holding the
; 16-byte x87 register image, and every operation on it is performed by a
; helper from compiler-rt.  This test makes sure each of those helpers is
; actually reachable from IR, so that a missing or misnamed libcall cannot slip
; through unnoticed.

target triple = "arm64ec-pc-windows-msvc"

declare x86_fp80 @llvm.fma.f80(x86_fp80, x86_fp80, x86_fp80)
declare x86_fp80 @llvm.maxnum.f80(x86_fp80, x86_fp80)
declare x86_fp80 @llvm.minnum.f80(x86_fp80, x86_fp80)
declare x86_fp80 @llvm.powi.f80.i32(x86_fp80, i32)
declare x86_fp80 @llvm.sqrt.f80(x86_fp80)
declare x86_fp80 @llvm.sin.f80(x86_fp80)
declare x86_fp80 @llvm.pow.f80(x86_fp80, x86_fp80)
declare x86_fp80 @llvm.floor.f80(x86_fp80)
declare x86_fp80 @llvm.experimental.constrained.fadd.f80(x86_fp80, x86_fp80, metadata, metadata)
declare fp128 @llvm.experimental.constrained.fpext.f128.f80(x86_fp80, metadata)
declare double @llvm.experimental.constrained.fptrunc.f64.f80(x86_fp80, metadata, metadata)

; Basic arithmetic.
; CHECK-LABEL: "#f_add":
; CHECK: bl "#__arm64ec_addxf3"
define x86_fp80 @f_add(x86_fp80 %a, x86_fp80 %b) {
  %r = fadd x86_fp80 %a, %b
  ret x86_fp80 %r
}

; CHECK-LABEL: "#f_sub":
; CHECK: bl "#__arm64ec_subxf3"
define x86_fp80 @f_sub(x86_fp80 %a, x86_fp80 %b) {
  %r = fsub x86_fp80 %a, %b
  ret x86_fp80 %r
}

; CHECK-LABEL: "#f_mul":
; CHECK: bl "#__arm64ec_mulxf3"
define x86_fp80 @f_mul(x86_fp80 %a, x86_fp80 %b) {
  %r = fmul x86_fp80 %a, %b
  ret x86_fp80 %r
}

; CHECK-LABEL: "#f_div":
; CHECK: bl "#__arm64ec_divxf3"
define x86_fp80 @f_div(x86_fp80 %a, x86_fp80 %b) {
  %r = fdiv x86_fp80 %a, %b
  ret x86_fp80 %r
}

; Three-operand helpers: the fused multiply-add lives in the "fmax" family of
; libgcc, the minimum and maximum in the "fmaxx"/"fminx" ones.
; CHECK-LABEL: "#f_fma":
; CHECK: bl "#__arm64ec_fmaxf3"
define x86_fp80 @f_fma(x86_fp80 %a, x86_fp80 %b, x86_fp80 %c) {
  %r = call x86_fp80 @llvm.fma.f80(x86_fp80 %a, x86_fp80 %b, x86_fp80 %c)
  ret x86_fp80 %r
}

; CHECK-LABEL: "#f_maxnum":
; CHECK: bl "#__arm64ec_fmaxxf3"
define x86_fp80 @f_maxnum(x86_fp80 %a, x86_fp80 %b) {
  %r = call x86_fp80 @llvm.maxnum.f80(x86_fp80 %a, x86_fp80 %b)
  ret x86_fp80 %r
}

; CHECK-LABEL: "#f_minnum":
; CHECK: bl "#__arm64ec_fminxf3"
define x86_fp80 @f_minnum(x86_fp80 %a, x86_fp80 %b) {
  %r = call x86_fp80 @llvm.minnum.f80(x86_fp80 %a, x86_fp80 %b)
  ret x86_fp80 %r
}

; CHECK-LABEL: "#f_powi":
; CHECK: bl "#__arm64ec_powixf2"
define x86_fp80 @f_powi(x86_fp80 %a, i32 %n) {
  %r = call x86_fp80 @llvm.powi.f80.i32(x86_fp80 %a, i32 %n)
  ret x86_fp80 %r
}

; The libm functions of the format come from the C runtime rather than from
; compiler-rt, and keep their unsuffixed names.
; CHECK-LABEL: "#libm_sqrt":
; CHECK: bl "#sqrtl"
define x86_fp80 @libm_sqrt(x86_fp80 %a) {
  %r = call x86_fp80 @llvm.sqrt.f80(x86_fp80 %a)
  ret x86_fp80 %r
}

; CHECK-LABEL: "#libm_sin":
; CHECK: bl "#sinl"
define x86_fp80 @libm_sin(x86_fp80 %a) {
  %r = call x86_fp80 @llvm.sin.f80(x86_fp80 %a)
  ret x86_fp80 %r
}

; CHECK-LABEL: "#libm_pow":
; CHECK: bl "#powl"
define x86_fp80 @libm_pow(x86_fp80 %a, x86_fp80 %b) {
  %r = call x86_fp80 @llvm.pow.f80(x86_fp80 %a, x86_fp80 %b)
  ret x86_fp80 %r
}

; CHECK-LABEL: "#libm_floor":
; CHECK: bl "#floorl"
define x86_fp80 @libm_floor(x86_fp80 %a) {
  %r = call x86_fp80 @llvm.floor.f80(x86_fp80 %a)
  ret x86_fp80 %r
}

; CHECK-LABEL: "#libm_fmod":
; CHECK: bl "#fmodl"
define x86_fp80 @libm_fmod(x86_fp80 %a, x86_fp80 %b) {
  %r = frem x86_fp80 %a, %b
  ret x86_fp80 %r
}

; Conversions from narrower and wider formats.  f16 has a helper that takes
; it directly, while bf16 has none and goes through f32.
; CHECK-LABEL: "#c_ext_half":
; CHECK: bl "#__arm64ec_extendhfxf2"
define x86_fp80 @c_ext_half(half %a) {
  %r = fpext half %a to x86_fp80
  ret x86_fp80 %r
}

; CHECK-LABEL: "#c_ext_float":
; CHECK: bl "#__arm64ec_extendsfxf2"
define x86_fp80 @c_ext_float(float %a) {
  %r = fpext float %a to x86_fp80
  ret x86_fp80 %r
}

; CHECK-LABEL: "#c_ext_double":
; CHECK: bl "#__arm64ec_extenddfxf2"
define x86_fp80 @c_ext_double(double %a) {
  %r = fpext double %a to x86_fp80
  ret x86_fp80 %r
}

; The source is softened here, not the destination.
; CHECK-LABEL: "#c_ext_f128":
; CHECK: bl "#__arm64ec_extendxftf2"
define fp128 @c_ext_f128(x86_fp80 %a) {
  %r = fpext x86_fp80 %a to fp128
  ret fp128 %r
}

; CHECK-LABEL: "#c_trunc_half":
; CHECK: bl "#__arm64ec_truncxfhf2"
define half @c_trunc_half(x86_fp80 %a) {
  %r = fptrunc x86_fp80 %a to half
  ret half %r
}

; CHECK-LABEL: "#c_trunc_bf16":
; CHECK: bl "#__arm64ec_truncxfbf2"
define bfloat @c_trunc_bf16(x86_fp80 %a) {
  %r = fptrunc x86_fp80 %a to bfloat
  ret bfloat %r
}

; CHECK-LABEL: "#c_trunc_float":
; CHECK: bl "#__arm64ec_truncxfsf2"
define float @c_trunc_float(x86_fp80 %a) {
  %r = fptrunc x86_fp80 %a to float
  ret float %r
}

; CHECK-LABEL: "#c_trunc_double":
; CHECK: bl "#__arm64ec_truncxfdf2"
define double @c_trunc_double(x86_fp80 %a) {
  %r = fptrunc x86_fp80 %a to double
  ret double %r
}

; CHECK-LABEL: "#c_trunc_f128":
; CHECK: bl "#__arm64ec_trunctfxf2"
define x86_fp80 @c_trunc_f128(fp128 %a) {
  %r = fptrunc fp128 %a to x86_fp80
  ret x86_fp80 %r
}

; Conversions to and from integers.
; CHECK-LABEL: "#i_fix_i32":
; CHECK: bl "#__arm64ec_fixxfsi"
define i32 @i_fix_i32(x86_fp80 %a) {
  %r = fptosi x86_fp80 %a to i32
  ret i32 %r
}

; CHECK-LABEL: "#i_fix_i64":
; CHECK: bl "#__arm64ec_fixxfdi"
define i64 @i_fix_i64(x86_fp80 %a) {
  %r = fptosi x86_fp80 %a to i64
  ret i64 %r
}

; CHECK-LABEL: "#i_fix_i128":
; CHECK: bl "#__arm64ec_fixxfti"
define i128 @i_fix_i128(x86_fp80 %a) {
  %r = fptosi x86_fp80 %a to i128
  ret i128 %r
}

; CHECK-LABEL: "#i_fixuns_i32":
; CHECK: bl "#__arm64ec_fixunsxfsi"
define i32 @i_fixuns_i32(x86_fp80 %a) {
  %r = fptoui x86_fp80 %a to i32
  ret i32 %r
}

; CHECK-LABEL: "#i_fixuns_i64":
; CHECK: bl "#__arm64ec_fixunsxfdi"
define i64 @i_fixuns_i64(x86_fp80 %a) {
  %r = fptoui x86_fp80 %a to i64
  ret i64 %r
}

; CHECK-LABEL: "#i_fixuns_i128":
; CHECK: bl "#__arm64ec_fixunsxfti"
define i128 @i_fixuns_i128(x86_fp80 %a) {
  %r = fptoui x86_fp80 %a to i128
  ret i128 %r
}

; CHECK-LABEL: "#i_float_i32":
; CHECK: bl "#__arm64ec_floatsixf"
define x86_fp80 @i_float_i32(i32 %a) {
  %r = sitofp i32 %a to x86_fp80
  ret x86_fp80 %r
}

; CHECK-LABEL: "#i_float_i64":
; CHECK: bl "#__arm64ec_floatdixf"
define x86_fp80 @i_float_i64(i64 %a) {
  %r = sitofp i64 %a to x86_fp80
  ret x86_fp80 %r
}

; CHECK-LABEL: "#i_float_i128":
; CHECK: bl "#__arm64ec_floattixf"
define x86_fp80 @i_float_i128(i128 %a) {
  %r = sitofp i128 %a to x86_fp80
  ret x86_fp80 %r
}

; CHECK-LABEL: "#i_floatuns_i32":
; CHECK: bl "#__arm64ec_floatunsixf"
define x86_fp80 @i_floatuns_i32(i32 %a) {
  %r = uitofp i32 %a to x86_fp80
  ret x86_fp80 %r
}

; CHECK-LABEL: "#i_floatuns_i64":
; CHECK: bl "#__arm64ec_floatundixf"
define x86_fp80 @i_floatuns_i64(i64 %a) {
  %r = uitofp i64 %a to x86_fp80
  ret x86_fp80 %r
}

; CHECK-LABEL: "#i_floatuns_i128":
; CHECK: bl "#__arm64ec_floatuntixf"
define x86_fp80 @i_floatuns_i128(i128 %a) {
  %r = uitofp i128 %a to x86_fp80
  ret x86_fp80 %r
}

; Every comparison predicate has its own helper.
; CHECK-LABEL: "#cmp_oeq":
; CHECK: bl "#__arm64ec_eqxf2"
define i1 @cmp_oeq(x86_fp80 %a, x86_fp80 %b) {
  %r = fcmp oeq x86_fp80 %a, %b
  ret i1 %r
}

; CHECK-LABEL: "#cmp_une":
; CHECK: bl "#__arm64ec_nexf2"
define i1 @cmp_une(x86_fp80 %a, x86_fp80 %b) {
  %r = fcmp une x86_fp80 %a, %b
  ret i1 %r
}

; CHECK-LABEL: "#cmp_oge":
; CHECK: bl "#__arm64ec_gexf2"
define i1 @cmp_oge(x86_fp80 %a, x86_fp80 %b) {
  %r = fcmp oge x86_fp80 %a, %b
  ret i1 %r
}

; CHECK-LABEL: "#cmp_olt":
; CHECK: bl "#__arm64ec_ltxf2"
define i1 @cmp_olt(x86_fp80 %a, x86_fp80 %b) {
  %r = fcmp olt x86_fp80 %a, %b
  ret i1 %r
}

; CHECK-LABEL: "#cmp_ole":
; CHECK: bl "#__arm64ec_lexf2"
define i1 @cmp_ole(x86_fp80 %a, x86_fp80 %b) {
  %r = fcmp ole x86_fp80 %a, %b
  ret i1 %r
}

; CHECK-LABEL: "#cmp_ogt":
; CHECK: bl "#__arm64ec_gtxf2"
define i1 @cmp_ogt(x86_fp80 %a, x86_fp80 %b) {
  %r = fcmp ogt x86_fp80 %a, %b
  ret i1 %r
}

; CHECK-LABEL: "#cmp_uno":
; CHECK: bl "#__arm64ec_unordxf2"
define i1 @cmp_uno(x86_fp80 %a, x86_fp80 %b) {
  %r = fcmp uno x86_fp80 %a, %b
  ret i1 %r
}

; The constrained intrinsics go through the same helpers.
; CHECK-LABEL: "#strict_fadd":
; CHECK: bl "#__arm64ec_addxf3"
define x86_fp80 @strict_fadd(x86_fp80 %a, x86_fp80 %b) strictfp {
  %r = call x86_fp80 @llvm.experimental.constrained.fadd.f80(
      x86_fp80 %a, x86_fp80 %b,
      metadata !"round.dynamic", metadata !"fpexcept.strict")
  ret x86_fp80 %r
}

; CHECK-LABEL: "#strict_fpext":
; CHECK: bl "#__arm64ec_extendxftf2"
define fp128 @strict_fpext(x86_fp80 %a) strictfp {
  %r = call fp128 @llvm.experimental.constrained.fpext.f128.f80(
      x86_fp80 %a, metadata !"fpexcept.strict")
  ret fp128 %r
}

; CHECK-LABEL: "#strict_fptrunc":
; CHECK: bl "#__arm64ec_truncxfdf2"
define double @strict_fptrunc(x86_fp80 %a) strictfp {
  %r = call double @llvm.experimental.constrained.fptrunc.f64.f80(
      x86_fp80 %a, metadata !"round.dynamic", metadata !"fpexcept.strict")
  ret double %r
}

