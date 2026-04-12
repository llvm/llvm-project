; RUN: llvm-as < %s | llvm-dis | FileCheck %s
;
; Test auto-upgrade of experimental.constrained.* intrinsics with non-default
; rounding modes and/or exception behaviors.  These produce new FP intrinsics
; with fp.control and/or fp.except operand bundles.
;
; Terminology:
;   Default  RM = round.dynamic   -> no fp.control bundle
;   Default  EB = fpexcept.strict -> no fp.except  bundle
;   Non-default RM (rte/rtz/rtp/rtn) -> fp.control bundle
;   Non-default EB (ignore/maytrap)  -> fp.except  bundle

; ===========================================================================
; 1.  Non-default rounding modes (all 4)
; ===========================================================================

; CHECK-LABEL: define float @rte(
; CHECK:         call float @llvm.fadd.f32(float %a, float %b) [ "fp.control"(metadata !"rte") ]
define float @rte(float %a, float %b) strictfp {
  %r = call float @llvm.experimental.constrained.fadd.f32(
      float %a, float %b, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret float %r
}

; CHECK-LABEL: define float @rtz(
; CHECK:         call float @llvm.fadd.f32(float %a, float %b) [ "fp.control"(metadata !"rtz") ]
define float @rtz(float %a, float %b) strictfp {
  %r = call float @llvm.experimental.constrained.fadd.f32(
      float %a, float %b, metadata !"round.towardzero", metadata !"fpexcept.strict")
  ret float %r
}

; CHECK-LABEL: define float @rtp(
; CHECK:         call float @llvm.fadd.f32(float %a, float %b) [ "fp.control"(metadata !"rtp") ]
define float @rtp(float %a, float %b) strictfp {
  %r = call float @llvm.experimental.constrained.fadd.f32(
      float %a, float %b, metadata !"round.upward", metadata !"fpexcept.strict")
  ret float %r
}

; CHECK-LABEL: define float @rtn(
; CHECK:         call float @llvm.fadd.f32(float %a, float %b) [ "fp.control"(metadata !"rtn") ]
define float @rtn(float %a, float %b) strictfp {
  %r = call float @llvm.experimental.constrained.fadd.f32(
      float %a, float %b, metadata !"round.downward", metadata !"fpexcept.strict")
  ret float %r
}

; ===========================================================================
; 2.  Non-default exception behaviors (ignore and maytrap)
; ===========================================================================

; CHECK-LABEL: define double @eb_ignore(
; CHECK:         call double @llvm.sqrt.f64(double %a) [ "fp.except"(metadata !"ignore") ]
define double @eb_ignore(double %a) strictfp {
  %r = call double @llvm.experimental.constrained.sqrt.f64(
      double %a, metadata !"round.dynamic", metadata !"fpexcept.ignore")
  ret double %r
}

; CHECK-LABEL: define double @eb_maytrap(
; CHECK:         call double @llvm.sqrt.f64(double %a) [ "fp.except"(metadata !"maytrap") ]
define double @eb_maytrap(double %a) strictfp {
  %r = call double @llvm.experimental.constrained.sqrt.f64(
      double %a, metadata !"round.dynamic", metadata !"fpexcept.maytrap")
  ret double %r
}

; ===========================================================================
; 3.  Both non-default RM and EB
; ===========================================================================

; CHECK-LABEL: define float @rte_maytrap(
; CHECK:         call float @llvm.fmul.f32(float %a, float %b) [ "fp.control"(metadata !"rte"), "fp.except"(metadata !"maytrap") ]
define float @rte_maytrap(float %a, float %b) strictfp {
  %r = call float @llvm.experimental.constrained.fmul.f32(
      float %a, float %b, metadata !"round.tonearest", metadata !"fpexcept.maytrap")
  ret float %r
}

; CHECK-LABEL: define float @rtz_ignore(
; CHECK:         call float @llvm.fdiv.f32(float %a, float %b) [ "fp.control"(metadata !"rtz"), "fp.except"(metadata !"ignore") ]
define float @rtz_ignore(float %a, float %b) strictfp {
  %r = call float @llvm.experimental.constrained.fdiv.f32(
      float %a, float %b, metadata !"round.towardzero", metadata !"fpexcept.ignore")
  ret float %r
}

; ===========================================================================
; 4.  Ops without rounding mode (fptoui, fpext, ceil) with non-default EB
; ===========================================================================

; CHECK-LABEL: define i32 @fptoui_maytrap(
; CHECK:         call i32 @llvm.fptoui.i32.f64(double %a) [ "fp.except"(metadata !"maytrap") ]
define i32 @fptoui_maytrap(double %a) strictfp {
  %r = call i32 @llvm.experimental.constrained.fptoui.i32.f64(
      double %a, metadata !"fpexcept.maytrap")
  ret i32 %r
}

; CHECK-LABEL: define double @fpext_ignore(
; CHECK:         call double @llvm.fpext.f64.f32(float %a) [ "fp.except"(metadata !"ignore") ]
define double @fpext_ignore(float %a) strictfp {
  %r = call double @llvm.experimental.constrained.fpext.f64.f32(
      float %a, metadata !"fpexcept.ignore")
  ret double %r
}

; CHECK-LABEL: define double @ceil_maytrap(
; CHECK:         call double @llvm.ceil.f64(double %a) [ "fp.except"(metadata !"maytrap") ]
define double @ceil_maytrap(double %a) strictfp {
  %r = call double @llvm.experimental.constrained.ceil.f64(
      double %a, metadata !"fpexcept.maytrap")
  ret double %r
}

; ===========================================================================
; 5.  Three-operand ops (fma / fmuladd) with non-default RM
; ===========================================================================

; CHECK-LABEL: define float @fma_rtp(
; CHECK:         call float @llvm.fma.f32(float %a, float %b, float %c) [ "fp.control"(metadata !"rtp") ]
define float @fma_rtp(float %a, float %b, float %c) strictfp {
  %r = call float @llvm.experimental.constrained.fma.f32(
      float %a, float %b, float %c, metadata !"round.upward", metadata !"fpexcept.strict")
  ret float %r
}

; CHECK-LABEL: define float @fmuladd_rtn(
; CHECK:         call float @llvm.fmuladd.f32(float %a, float %b, float %c) [ "fp.control"(metadata !"rtn") ]
define float @fmuladd_rtn(float %a, float %b, float %c) strictfp {
  %r = call float @llvm.experimental.constrained.fmuladd.f32(
      float %a, float %b, float %c, metadata !"round.downward", metadata !"fpexcept.strict")
  ret float %r
}

; ===========================================================================
; 6.  fcmp with non-default EB and all predicates
;     (no RM for fcmp; EB non-default → fp.except bundle)
; ===========================================================================

; CHECK-LABEL: define i1 @fcmp_oeq_maytrap(
; CHECK:         call i1 @llvm.fcmp.f32(float %a, float %b, metadata !"oeq") [ "fp.except"(metadata !"maytrap") ]
define i1 @fcmp_oeq_maytrap(float %a, float %b) strictfp {
  %r = call i1 @llvm.experimental.constrained.fcmp.f32(
      float %a, float %b, metadata !"oeq", metadata !"fpexcept.maytrap")
  ret i1 %r
}

; CHECK-LABEL: define i1 @fcmp_ogt_ignore(
; CHECK:         call i1 @llvm.fcmp.f32(float %a, float %b, metadata !"ogt") [ "fp.except"(metadata !"ignore") ]
define i1 @fcmp_ogt_ignore(float %a, float %b) strictfp {
  %r = call i1 @llvm.experimental.constrained.fcmp.f32(
      float %a, float %b, metadata !"ogt", metadata !"fpexcept.ignore")
  ret i1 %r
}

; CHECK-LABEL: define i1 @fcmp_one_maytrap(
; CHECK:         call i1 @llvm.fcmp.f32(float %a, float %b, metadata !"one") [ "fp.except"(metadata !"maytrap") ]
define i1 @fcmp_one_maytrap(float %a, float %b) strictfp {
  %r = call i1 @llvm.experimental.constrained.fcmp.f32(
      float %a, float %b, metadata !"one", metadata !"fpexcept.maytrap")
  ret i1 %r
}

; CHECK-LABEL: define i1 @fcmp_ult_maytrap(
; CHECK:         call i1 @llvm.fcmp.f32(float %a, float %b, metadata !"ult") [ "fp.except"(metadata !"maytrap") ]
define i1 @fcmp_ult_maytrap(float %a, float %b) strictfp {
  %r = call i1 @llvm.experimental.constrained.fcmp.f32(
      float %a, float %b, metadata !"ult", metadata !"fpexcept.maytrap")
  ret i1 %r
}

; CHECK-LABEL: define i1 @fcmp_uno_maytrap(
; CHECK:         call i1 @llvm.fcmp.f32(float %a, float %b, metadata !"uno") [ "fp.except"(metadata !"maytrap") ]
define i1 @fcmp_uno_maytrap(float %a, float %b) strictfp {
  %r = call i1 @llvm.experimental.constrained.fcmp.f32(
      float %a, float %b, metadata !"uno", metadata !"fpexcept.maytrap")
  ret i1 %r
}

; ===========================================================================
; 7.  Conversion ops with non-default RM
; ===========================================================================

; CHECK-LABEL: define double @sitofp_rte(
; CHECK:         call double @llvm.sitofp.f64.i32(i32 %i) [ "fp.control"(metadata !"rte") ]
define double @sitofp_rte(i32 %i) strictfp {
  %r = call double @llvm.experimental.constrained.sitofp.f64.i32(
      i32 %i, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret double %r
}

; CHECK-LABEL: define float @fptrunc_rtz(
; CHECK:         call float @llvm.fptrunc.f32.f64(double %a) [ "fp.control"(metadata !"rtz") ]
define float @fptrunc_rtz(double %a) strictfp {
  %r = call float @llvm.experimental.constrained.fptrunc.f32.f64(
      double %a, metadata !"round.towardzero", metadata !"fpexcept.strict")
  ret float %r
}

; CHECK-LABEL: define i64 @lrint_rtn(
; CHECK:         call i64 @llvm.lrint.i64.f64(double %a) [ "fp.control"(metadata !"rtn") ]
define i64 @lrint_rtn(double %a) strictfp {
  %r = call i64 @llvm.experimental.constrained.lrint.i64.f64(
      double %a, metadata !"round.downward", metadata !"fpexcept.strict")
  ret i64 %r
}

; CHECK-LABEL: define i64 @llrint_rtp(
; CHECK:         call i64 @llvm.llrint.i64.f64(double %a) [ "fp.control"(metadata !"rtp") ]
define i64 @llrint_rtp(double %a) strictfp {
  %r = call i64 @llvm.experimental.constrained.llrint.i64.f64(
      double %a, metadata !"round.upward", metadata !"fpexcept.strict")
  ret i64 %r
}

; ===========================================================================
; 8.  ldexp (two-type overload) and powi (int exponent) with non-default RM
; ===========================================================================

; CHECK-LABEL: define double @ldexp_rte(
; CHECK:         call double @llvm.ldexp.f64.i32(double %a, i32 %n) [ "fp.control"(metadata !"rte") ]
define double @ldexp_rte(double %a, i32 %n) strictfp {
  %r = call double @llvm.experimental.constrained.ldexp.f64(
      double %a, i32 %n, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret double %r
}

; CHECK-LABEL: define double @powi_rtz(
; CHECK:         call double @llvm.powi.f64.i32(double %a, i32 %n) [ "fp.control"(metadata !"rtz") ]
define double @powi_rtz(double %a, i32 %n) strictfp {
  %r = call double @llvm.experimental.constrained.powi.f64(
      double %a, i32 %n, metadata !"round.towardzero", metadata !"fpexcept.strict")
  ret double %r
}

; ===========================================================================
; Declarations required for the old-form intrinsics used above
; ===========================================================================

declare float @llvm.experimental.constrained.fadd.f32(float, float, metadata, metadata)
declare float @llvm.experimental.constrained.fmul.f32(float, float, metadata, metadata)
declare float @llvm.experimental.constrained.fdiv.f32(float, float, metadata, metadata)
declare float @llvm.experimental.constrained.fma.f32(float, float, float, metadata, metadata)
declare float @llvm.experimental.constrained.fmuladd.f32(float, float, float, metadata, metadata)
declare double @llvm.experimental.constrained.sqrt.f64(double, metadata, metadata)
declare i32 @llvm.experimental.constrained.fptoui.i32.f64(double, metadata)
declare double @llvm.experimental.constrained.fpext.f64.f32(float, metadata)
declare double @llvm.experimental.constrained.ceil.f64(double, metadata)
declare double @llvm.experimental.constrained.sitofp.f64.i32(i32, metadata, metadata)
declare float @llvm.experimental.constrained.fptrunc.f32.f64(double, metadata, metadata)
declare i64 @llvm.experimental.constrained.lrint.i64.f64(double, metadata, metadata)
declare i64 @llvm.experimental.constrained.llrint.i64.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.ldexp.f64(double, i32, metadata, metadata)
declare double @llvm.experimental.constrained.powi.f64(double, i32, metadata, metadata)
declare i1 @llvm.experimental.constrained.fcmp.f32(float, float, metadata, metadata)
