; RUN: llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_80 -debug-only=isel -o /dev/null < %s 2>&1 | FileCheck %s
; REQUIRES: asserts

; Verify that SDNode dumps include FP environment flags set by fp.control
; operand bundles (denorm modes and rounding modes).  These flags live in
; SDNodeFlags and are printed after any fast-math flags.
;
; Key observations:
; - IEEE (the default denorm mode) is NOT printed — FlagInserter propagates
;   it to every FP node, so printing it would be extremely noisy.
; - Non-default denorm modes (zero, pzero) are printed.
; - F32-specific flags are printed only when the f32 pair differs from the
;   general pair.
; - Explicit rounding modes (from fp.control bundles) are always printed.

declare float @llvm.fadd.f32(float, float)
declare double @llvm.fadd.f64(double, double)
declare float @llvm.fdiv.f32(float, float)
declare float @llvm.frem.f32(float, float)
declare float @llvm.fptrunc.f32.f64(double)
declare double @llvm.fpext.f64.f32(float)
declare i1 @llvm.fcmp.f32(float, float, metadata)

; No bundle: function-attribute default (IEEE) is NOT printed (suppressed to avoid noise).
; CHECK-LABEL: Initial selection DAG: %bb.0 'no_bundle:'
; CHECK: fadd
; CHECK-NOT: ieee
define float @no_bundle(float %a, float %b) {
  %r = call float @llvm.fadd.f32(float %a, float %b)
  ret float %r
}

; Rounding mode only (rtz): RM appended after denorm flags.
; CHECK-LABEL: Initial selection DAG: %bb.0 'rm_only:'
; CHECK: fadd{{.*}} rtz
define float @rm_only(float %a, float %b) {
  %r = call float @llvm.fadd.f32(float %a, float %b) [ "fp.control"(metadata !"rtz") ]
  ret float %r
}

; General denorm in==out (PreserveSign): prints "zero".
; CHECK-LABEL: Initial selection DAG: %bb.0 'denorm_in_eq_out:'
; CHECK: fadd zero
define float @denorm_in_eq_out(float %a, float %b) {
  %r = call float @llvm.fadd.f32(float %a, float %b) [ "fp.control"(metadata !"denorm.in=zero", metadata !"denorm.out=zero") ]
  ret float %r
}

; General denorm in!=out: prints "zero:ieee".
; CHECK-LABEL: Initial selection DAG: %bb.0 'denorm_in_ne_out:'
; CHECK: fadd zero:ieee
define float @denorm_in_ne_out(float %a, float %b) {
  %r = call float @llvm.fadd.f32(float %a, float %b) [ "fp.control"(metadata !"denorm.in=zero", metadata !"denorm.out=ieee") ]
  ret float %r
}

; F32-specific denorm on an f64 op: general stays IEEE (default, suppressed),
; f32 is zero. Only the f32 deviation is printed: "zero.f32".
; CHECK-LABEL: Initial selection DAG: %bb.0 'f32_denorm_on_f64:'
; CHECK: fadd zero.f32
define double @f32_denorm_on_f64(double %a, double %b) {
  %r = call double @llvm.fadd.f64(double %a, double %b) [ "fp.control"(metadata !"denorm.f32.in=zero", metadata !"denorm.f32.out=zero") ]
  ret double %r
}

; F32-specific denorm in!=out on f64 op: general is IEEE (default, suppressed),
; f32 pair differs from general → "zero.f32:pzero.f32".
; CHECK-LABEL: Initial selection DAG: %bb.0 'f32_denorm_ne_on_f64:'
; CHECK: fadd zero.f32:pzero.f32
define double @f32_denorm_ne_on_f64(double %a, double %b) {
  %r = call double @llvm.fadd.f64(double %a, double %b) [ "fp.control"(metadata !"denorm.f32.in=zero", metadata !"denorm.f32.out=pzero") ]
  ret double %r
}

; General + F32 same pair (f64 op): general=zero, f32=zero → f32 suppressed.
; CHECK-LABEL: Initial selection DAG: %bb.0 'general_and_f32_same_on_f64:'
; CHECK: fadd zero
; CHECK-NOT: .f32
define double @general_and_f32_same_on_f64(double %a, double %b) {
  %r = call double @llvm.fadd.f64(double %a, double %b) [ "fp.control"(metadata !"denorm.in=zero", metadata !"denorm.out=zero", metadata !"denorm.f32.in=zero", metadata !"denorm.f32.out=zero") ]
  ret double %r
}

; General + F32 different pairs (f64 op): general=zero, f32=zero:pzero.
; Prints "zero zero.f32:pzero.f32".
; CHECK-LABEL: Initial selection DAG: %bb.0 'general_and_f32_differ_on_f64:'
; CHECK: fadd zero zero.f32:pzero.f32
define double @general_and_f32_differ_on_f64(double %a, double %b) {
  %r = call double @llvm.fadd.f64(double %a, double %b) [ "fp.control"(metadata !"denorm.in=zero", metadata !"denorm.out=zero", metadata !"denorm.f32.in=zero", metadata !"denorm.f32.out=pzero") ]
  ret double %r
}

; All three: general denorm + f32 denorm + RM (f64 op).
; Prints "zero zero.f32:ieee.f32 rtz".
; CHECK-LABEL: Initial selection DAG: %bb.0 'all_three_on_f64:'
; CHECK: fadd zero zero.f32:ieee.f32 rtz
define double @all_three_on_f64(double %a, double %b) {
  %r = call double @llvm.fadd.f64(double %a, double %b) [ "fp.control"(metadata !"denorm.in=zero", metadata !"denorm.out=zero", metadata !"denorm.f32.in=zero", metadata !"denorm.f32.out=ieee", metadata !"rtz") ]
  ret double %r
}

; fdiv with FTZ bundle.
; CHECK-LABEL: Initial selection DAG: %bb.0 'fdiv_ftz:'
; CHECK: fdiv zero
define float @fdiv_ftz(float %a, float %b) {
  %r = call float @llvm.fdiv.f32(float %a, float %b)
       [ "fp.control"(metadata !"denorm.in=zero", metadata !"denorm.out=zero") ]
  ret float %r
}

; frem with rtz bundle.
; CHECK-LABEL: Initial selection DAG: %bb.0 'frem_rtz:'
; CHECK: frem{{.*}} rtz
define float @frem_rtz(float %a, float %b) {
  %r = call float @llvm.frem.f32(float %a, float %b)
       [ "fp.control"(metadata !"rtz") ]
  ret float %r
}

; fptrunc (f64->f32) with rtz bundle.
; CHECK-LABEL: Initial selection DAG: %bb.0 'fptrunc_rtz:'
; CHECK: fp_round{{.*}} rtz
define float @fptrunc_rtz(double %a) {
  %r = call float @llvm.fptrunc.f32.f64(double %a)
       [ "fp.control"(metadata !"rtz") ]
  ret float %r
}

; fpext (f32->f64): exact, no rounding — IEEE default is suppressed, no flags expected.
; CHECK-LABEL: Initial selection DAG: %bb.0 'fpext_default:'
; CHECK: fp_extend
; CHECK-NOT: ieee
define double @fpext_default(float %a) {
  %r = call double @llvm.fpext.f64.f32(float %a)
  ret double %r
}

; fcmp oeq with FTZ bundle.
; CHECK-LABEL: Initial selection DAG: %bb.0 'fcmp_oeq_ftz:'
; CHECK: setcc{{.*}} zero
define i1 @fcmp_oeq_ftz(float %a, float %b) {
  %r = call i1 @llvm.fcmp.f32(float %a, float %b, metadata !"oeq")
       [ "fp.control"(metadata !"denorm.in=zero", metadata !"denorm.out=zero") ]
  ret i1 %r
}
