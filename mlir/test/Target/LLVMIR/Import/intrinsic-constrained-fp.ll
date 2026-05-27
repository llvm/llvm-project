; RUN: mlir-translate -import-llvm %s -split-input-file | FileCheck %s

; -----------------------------------------------------------------------------
; FP -> int conversions: one fp arg, single exception-behavior metadata.
; -----------------------------------------------------------------------------

declare i32 @llvm.experimental.constrained.fptosi.i32.f32(float, metadata)

; CHECK-LABEL: llvm.func @constrained_fptosi
; CHECK: %[[EB:.*]] = llvm.mlir.metadata_as_value #llvm.md_string<"fpexcept.strict">
; CHECK: %{{.*}} = llvm.call_intrinsic "llvm.experimental.constrained.fptosi.i32.f32"(%{{.*}}, %[[EB]]) : (f32, !llvm.metadata) -> i32
define i32 @constrained_fptosi(float %a) {
  %r = call i32 @llvm.experimental.constrained.fptosi.i32.f32(float %a, metadata !"fpexcept.strict")
  ret i32 %r
}

; // -----

declare i32 @llvm.experimental.constrained.fptoui.i32.f32(float, metadata)

; CHECK-LABEL: llvm.func @constrained_fptoui
; CHECK: %[[EB:.*]] = llvm.mlir.metadata_as_value #llvm.md_string<"fpexcept.ignore">
; CHECK: %{{.*}} = llvm.call_intrinsic "llvm.experimental.constrained.fptoui.i32.f32"(%{{.*}}, %[[EB]]) : (f32, !llvm.metadata) -> i32
define i32 @constrained_fptoui(float %a) {
  %r = call i32 @llvm.experimental.constrained.fptoui.i32.f32(float %a, metadata !"fpexcept.ignore")
  ret i32 %r
}

; -----------------------------------------------------------------------------
; Unary math intrinsics: one fp arg, rounding-mode + exception-behavior.
; -----------------------------------------------------------------------------

; // -----

declare float @llvm.experimental.constrained.sqrt.f32(float, metadata, metadata)

; CHECK-LABEL: llvm.func @constrained_sqrt
; CHECK: %[[RM:.*]] = llvm.mlir.metadata_as_value #llvm.md_string<"round.tonearest">
; CHECK: %[[EB:.*]] = llvm.mlir.metadata_as_value #llvm.md_string<"fpexcept.strict">
; CHECK: %{{.*}} = llvm.call_intrinsic "llvm.experimental.constrained.sqrt.f32"(%{{.*}}, %[[RM]], %[[EB]]) : (f32, !llvm.metadata, !llvm.metadata) -> f32
define float @constrained_sqrt(float %a) {
  %r = call float @llvm.experimental.constrained.sqrt.f32(float %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret float %r
}

; // -----

declare float @llvm.experimental.constrained.asin.f32(float, metadata, metadata)

; CHECK-LABEL: llvm.func @constrained_asin
; CHECK: %[[RM:.*]] = llvm.mlir.metadata_as_value #llvm.md_string<"round.tonearest">
; CHECK: %[[EB:.*]] = llvm.mlir.metadata_as_value #llvm.md_string<"fpexcept.strict">
; CHECK: %{{.*}} = llvm.call_intrinsic "llvm.experimental.constrained.asin.f32"(%{{.*}}, %[[RM]], %[[EB]]) : (f32, !llvm.metadata, !llvm.metadata) -> f32
define float @constrained_asin(float %a) {
  %r = call float @llvm.experimental.constrained.asin.f32(float %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret float %r
}

; // -----

declare float @llvm.experimental.constrained.acos.f32(float, metadata, metadata)

; CHECK-LABEL: llvm.func @constrained_acos
; CHECK: llvm.mlir.metadata_as_value #llvm.md_string<"round.towardzero">
; CHECK: llvm.mlir.metadata_as_value #llvm.md_string<"fpexcept.maytrap">
; CHECK: llvm.call_intrinsic "llvm.experimental.constrained.acos.f32"
define float @constrained_acos(float %a) {
  %r = call float @llvm.experimental.constrained.acos.f32(float %a, metadata !"round.towardzero", metadata !"fpexcept.maytrap")
  ret float %r
}

; // -----

declare float @llvm.experimental.constrained.atan.f32(float, metadata, metadata)

; CHECK-LABEL: llvm.func @constrained_atan
; CHECK: llvm.mlir.metadata_as_value #llvm.md_string<"round.upward">
; CHECK: llvm.mlir.metadata_as_value #llvm.md_string<"fpexcept.ignore">
; CHECK: llvm.call_intrinsic "llvm.experimental.constrained.atan.f32"
define float @constrained_atan(float %a) {
  %r = call float @llvm.experimental.constrained.atan.f32(float %a, metadata !"round.upward", metadata !"fpexcept.ignore")
  ret float %r
}

; // -----

declare float @llvm.experimental.constrained.sin.f32(float, metadata, metadata)

; CHECK-LABEL: llvm.func @constrained_sin
; CHECK: llvm.mlir.metadata_as_value #llvm.md_string<"round.downward">
; CHECK: llvm.mlir.metadata_as_value #llvm.md_string<"fpexcept.strict">
; CHECK: llvm.call_intrinsic "llvm.experimental.constrained.sin.f32"
define float @constrained_sin(float %a) {
  %r = call float @llvm.experimental.constrained.sin.f32(float %a, metadata !"round.downward", metadata !"fpexcept.strict")
  ret float %r
}

; // -----

declare float @llvm.experimental.constrained.cos.f32(float, metadata, metadata)

; CHECK-LABEL: llvm.func @constrained_cos
; CHECK: llvm.mlir.metadata_as_value #llvm.md_string<"round.tonearestaway">
; CHECK: llvm.mlir.metadata_as_value #llvm.md_string<"fpexcept.ignore">
; CHECK: llvm.call_intrinsic "llvm.experimental.constrained.cos.f32"
define float @constrained_cos(float %a) {
  %r = call float @llvm.experimental.constrained.cos.f32(float %a, metadata !"round.tonearestaway", metadata !"fpexcept.ignore")
  ret float %r
}

; // -----

declare float @llvm.experimental.constrained.tan.f32(float, metadata, metadata)

; CHECK-LABEL: llvm.func @constrained_tan
; CHECK: llvm.mlir.metadata_as_value #llvm.md_string<"round.dynamic">
; CHECK: llvm.mlir.metadata_as_value #llvm.md_string<"fpexcept.strict">
; CHECK: llvm.call_intrinsic "llvm.experimental.constrained.tan.f32"
define float @constrained_tan(float %a) {
  %r = call float @llvm.experimental.constrained.tan.f32(float %a, metadata !"round.dynamic", metadata !"fpexcept.strict")
  ret float %r
}

; // -----

declare float @llvm.experimental.constrained.sinh.f32(float, metadata, metadata)

; CHECK-LABEL: llvm.func @constrained_sinh
; CHECK: llvm.mlir.metadata_as_value #llvm.md_string<"round.tonearest">
; CHECK: llvm.mlir.metadata_as_value #llvm.md_string<"fpexcept.strict">
; CHECK: llvm.call_intrinsic "llvm.experimental.constrained.sinh.f32"
define float @constrained_sinh(float %a) {
  %r = call float @llvm.experimental.constrained.sinh.f32(float %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret float %r
}

; // -----

declare float @llvm.experimental.constrained.cosh.f32(float, metadata, metadata)

; CHECK-LABEL: llvm.func @constrained_cosh
; CHECK: llvm.mlir.metadata_as_value #llvm.md_string<"round.tonearest">
; CHECK: llvm.mlir.metadata_as_value #llvm.md_string<"fpexcept.strict">
; CHECK: llvm.call_intrinsic "llvm.experimental.constrained.cosh.f32"
define float @constrained_cosh(float %a) {
  %r = call float @llvm.experimental.constrained.cosh.f32(float %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret float %r
}

; // -----

declare float @llvm.experimental.constrained.tanh.f32(float, metadata, metadata)

; CHECK-LABEL: llvm.func @constrained_tanh
; CHECK: llvm.mlir.metadata_as_value #llvm.md_string<"round.tonearest">
; CHECK: llvm.mlir.metadata_as_value #llvm.md_string<"fpexcept.strict">
; CHECK: llvm.call_intrinsic "llvm.experimental.constrained.tanh.f32"
define float @constrained_tanh(float %a) {
  %r = call float @llvm.experimental.constrained.tanh.f32(float %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret float %r
}

; // -----

declare float @llvm.experimental.constrained.log.f32(float, metadata, metadata)

; CHECK-LABEL: llvm.func @constrained_log
; CHECK: llvm.mlir.metadata_as_value #llvm.md_string<"round.tonearest">
; CHECK: llvm.mlir.metadata_as_value #llvm.md_string<"fpexcept.strict">
; CHECK: llvm.call_intrinsic "llvm.experimental.constrained.log.f32"
define float @constrained_log(float %a) {
  %r = call float @llvm.experimental.constrained.log.f32(float %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret float %r
}

; // -----

declare float @llvm.experimental.constrained.log10.f32(float, metadata, metadata)

; CHECK-LABEL: llvm.func @constrained_log10
; CHECK: llvm.mlir.metadata_as_value #llvm.md_string<"round.tonearest">
; CHECK: llvm.mlir.metadata_as_value #llvm.md_string<"fpexcept.strict">
; CHECK: llvm.call_intrinsic "llvm.experimental.constrained.log10.f32"
define float @constrained_log10(float %a) {
  %r = call float @llvm.experimental.constrained.log10.f32(float %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret float %r
}

; // -----

declare float @llvm.experimental.constrained.log2.f32(float, metadata, metadata)

; CHECK-LABEL: llvm.func @constrained_log2
; CHECK: llvm.mlir.metadata_as_value #llvm.md_string<"round.tonearest">
; CHECK: llvm.mlir.metadata_as_value #llvm.md_string<"fpexcept.strict">
; CHECK: llvm.call_intrinsic "llvm.experimental.constrained.log2.f32"
define float @constrained_log2(float %a) {
  %r = call float @llvm.experimental.constrained.log2.f32(float %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret float %r
}

; // -----

declare float @llvm.experimental.constrained.exp.f32(float, metadata, metadata)

; CHECK-LABEL: llvm.func @constrained_exp
; CHECK: llvm.mlir.metadata_as_value #llvm.md_string<"round.tonearest">
; CHECK: llvm.mlir.metadata_as_value #llvm.md_string<"fpexcept.strict">
; CHECK: llvm.call_intrinsic "llvm.experimental.constrained.exp.f32"
define float @constrained_exp(float %a) {
  %r = call float @llvm.experimental.constrained.exp.f32(float %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret float %r
}

; // -----

declare float @llvm.experimental.constrained.exp2.f32(float, metadata, metadata)

; CHECK-LABEL: llvm.func @constrained_exp2
; CHECK: llvm.mlir.metadata_as_value #llvm.md_string<"round.tonearest">
; CHECK: llvm.mlir.metadata_as_value #llvm.md_string<"fpexcept.strict">
; CHECK: llvm.call_intrinsic "llvm.experimental.constrained.exp2.f32"
define float @constrained_exp2(float %a) {
  %r = call float @llvm.experimental.constrained.exp2.f32(float %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret float %r
}

; // -----

declare float @llvm.experimental.constrained.rint.f32(float, metadata, metadata)

; CHECK-LABEL: llvm.func @constrained_rint
; CHECK: llvm.mlir.metadata_as_value #llvm.md_string<"round.upward">
; CHECK: llvm.mlir.metadata_as_value #llvm.md_string<"fpexcept.strict">
; CHECK: llvm.call_intrinsic "llvm.experimental.constrained.rint.f32"
define float @constrained_rint(float %a) {
  %r = call float @llvm.experimental.constrained.rint.f32(float %a, metadata !"round.upward", metadata !"fpexcept.strict")
  ret float %r
}

; // -----

declare float @llvm.experimental.constrained.nearbyint.f32(float, metadata, metadata)

; CHECK-LABEL: llvm.func @constrained_nearbyint
; CHECK: llvm.mlir.metadata_as_value #llvm.md_string<"round.downward">
; CHECK: llvm.mlir.metadata_as_value #llvm.md_string<"fpexcept.ignore">
; CHECK: llvm.call_intrinsic "llvm.experimental.constrained.nearbyint.f32"
define float @constrained_nearbyint(float %a) {
  %r = call float @llvm.experimental.constrained.nearbyint.f32(float %a, metadata !"round.downward", metadata !"fpexcept.ignore")
  ret float %r
}

; -----------------------------------------------------------------------------
; Binary math intrinsics: two fp args, rounding + exception metadata.
; -----------------------------------------------------------------------------

; // -----

declare float @llvm.experimental.constrained.atan2.f32(float, float, metadata, metadata)

; CHECK-LABEL: llvm.func @constrained_atan2
; CHECK: %[[RM:.*]] = llvm.mlir.metadata_as_value #llvm.md_string<"round.tonearest">
; CHECK: %[[EB:.*]] = llvm.mlir.metadata_as_value #llvm.md_string<"fpexcept.strict">
; CHECK: %{{.*}} = llvm.call_intrinsic "llvm.experimental.constrained.atan2.f32"(%{{.*}}, %{{.*}}, %[[RM]], %[[EB]]) : (f32, f32, !llvm.metadata, !llvm.metadata) -> f32
define float @constrained_atan2(float %a, float %b) {
  %r = call float @llvm.experimental.constrained.atan2.f32(float %a, float %b, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret float %r
}

; // -----

declare float @llvm.experimental.constrained.pow.f32(float, float, metadata, metadata)

; CHECK-LABEL: llvm.func @constrained_pow
; CHECK: llvm.mlir.metadata_as_value #llvm.md_string<"round.tonearest">
; CHECK: llvm.mlir.metadata_as_value #llvm.md_string<"fpexcept.strict">
; CHECK: llvm.call_intrinsic "llvm.experimental.constrained.pow.f32"
define float @constrained_pow(float %a, float %b) {
  %r = call float @llvm.experimental.constrained.pow.f32(float %a, float %b, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret float %r
}

; -----------------------------------------------------------------------------
; Mixed-type binary intrinsics.
; -----------------------------------------------------------------------------

; // -----

declare float @llvm.experimental.constrained.powi.f32(float, i32, metadata, metadata)

; CHECK-LABEL: llvm.func @constrained_powi
; CHECK: %[[RM:.*]] = llvm.mlir.metadata_as_value #llvm.md_string<"round.tonearest">
; CHECK: %[[EB:.*]] = llvm.mlir.metadata_as_value #llvm.md_string<"fpexcept.strict">
; CHECK: %{{.*}} = llvm.call_intrinsic "llvm.experimental.constrained.powi.f32"(%{{.*}}, %{{.*}}, %[[RM]], %[[EB]]) : (f32, i32, !llvm.metadata, !llvm.metadata) -> f32
define float @constrained_powi(float %a, i32 %b) {
  %r = call float @llvm.experimental.constrained.powi.f32(float %a, i32 %b, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret float %r
}

; // -----

declare float @llvm.experimental.constrained.ldexp.f32.i32(float, i32, metadata, metadata)

; CHECK-LABEL: llvm.func @constrained_ldexp
; CHECK: %[[RM:.*]] = llvm.mlir.metadata_as_value #llvm.md_string<"round.tonearest">
; CHECK: %[[EB:.*]] = llvm.mlir.metadata_as_value #llvm.md_string<"fpexcept.strict">
; CHECK: %{{.*}} = llvm.call_intrinsic "llvm.experimental.constrained.ldexp.f32.i32"(%{{.*}}, %{{.*}}, %[[RM]], %[[EB]]) : (f32, i32, !llvm.metadata, !llvm.metadata) -> f32
define float @constrained_ldexp(float %a, i32 %b) {
  %r = call float @llvm.experimental.constrained.ldexp.f32.i32(float %a, i32 %b, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret float %r
}

; -----------------------------------------------------------------------------
; FP -> int rounding (lrint / llrint): rounding + exception metadata.
; -----------------------------------------------------------------------------

; // -----

declare i32 @llvm.experimental.constrained.lrint.i32.f32(float, metadata, metadata)

; CHECK-LABEL: llvm.func @constrained_lrint
; CHECK: %[[RM:.*]] = llvm.mlir.metadata_as_value #llvm.md_string<"round.tonearest">
; CHECK: %[[EB:.*]] = llvm.mlir.metadata_as_value #llvm.md_string<"fpexcept.strict">
; CHECK: %{{.*}} = llvm.call_intrinsic "llvm.experimental.constrained.lrint.i32.f32"(%{{.*}}, %[[RM]], %[[EB]]) : (f32, !llvm.metadata, !llvm.metadata) -> i32
define i32 @constrained_lrint(float %a) {
  %r = call i32 @llvm.experimental.constrained.lrint.i32.f32(float %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret i32 %r
}

; // -----

declare i64 @llvm.experimental.constrained.llrint.i64.f32(float, metadata, metadata)

; CHECK-LABEL: llvm.func @constrained_llrint
; CHECK: llvm.mlir.metadata_as_value #llvm.md_string<"round.tonearest">
; CHECK: llvm.mlir.metadata_as_value #llvm.md_string<"fpexcept.strict">
; CHECK: llvm.call_intrinsic "llvm.experimental.constrained.llrint.i64.f32"
define i64 @constrained_llrint(float %a) {
  %r = call i64 @llvm.experimental.constrained.llrint.i64.f32(float %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret i64 %r
}

; -----------------------------------------------------------------------------
; Binary intrinsics with exception-behavior metadata only (no rounding mode).
; -----------------------------------------------------------------------------

; // -----

declare float @llvm.experimental.constrained.maxnum.f32(float, float, metadata)

; CHECK-LABEL: llvm.func @constrained_maxnum
; CHECK: %[[EB:.*]] = llvm.mlir.metadata_as_value #llvm.md_string<"fpexcept.strict">
; CHECK: %{{.*}} = llvm.call_intrinsic "llvm.experimental.constrained.maxnum.f32"(%{{.*}}, %{{.*}}, %[[EB]]) : (f32, f32, !llvm.metadata) -> f32
define float @constrained_maxnum(float %a, float %b) {
  %r = call float @llvm.experimental.constrained.maxnum.f32(float %a, float %b, metadata !"fpexcept.strict")
  ret float %r
}

; // -----

declare float @llvm.experimental.constrained.minnum.f32(float, float, metadata)

; CHECK-LABEL: llvm.func @constrained_minnum
; CHECK: llvm.mlir.metadata_as_value #llvm.md_string<"fpexcept.ignore">
; CHECK: llvm.call_intrinsic "llvm.experimental.constrained.minnum.f32"
define float @constrained_minnum(float %a, float %b) {
  %r = call float @llvm.experimental.constrained.minnum.f32(float %a, float %b, metadata !"fpexcept.ignore")
  ret float %r
}

; // -----

declare float @llvm.experimental.constrained.maximum.f32(float, float, metadata)

; CHECK-LABEL: llvm.func @constrained_maximum
; CHECK: llvm.mlir.metadata_as_value #llvm.md_string<"fpexcept.maytrap">
; CHECK: llvm.call_intrinsic "llvm.experimental.constrained.maximum.f32"
define float @constrained_maximum(float %a, float %b) {
  %r = call float @llvm.experimental.constrained.maximum.f32(float %a, float %b, metadata !"fpexcept.maytrap")
  ret float %r
}

; // -----

declare float @llvm.experimental.constrained.minimum.f32(float, float, metadata)

; CHECK-LABEL: llvm.func @constrained_minimum
; CHECK: llvm.mlir.metadata_as_value #llvm.md_string<"fpexcept.strict">
; CHECK: llvm.call_intrinsic "llvm.experimental.constrained.minimum.f32"
define float @constrained_minimum(float %a, float %b) {
  %r = call float @llvm.experimental.constrained.minimum.f32(float %a, float %b, metadata !"fpexcept.strict")
  ret float %r
}

; -----------------------------------------------------------------------------
; Unary intrinsics with exception-behavior metadata only (no rounding mode).
; -----------------------------------------------------------------------------

; // -----

declare float @llvm.experimental.constrained.ceil.f32(float, metadata)

; CHECK-LABEL: llvm.func @constrained_ceil
; CHECK: %[[EB:.*]] = llvm.mlir.metadata_as_value #llvm.md_string<"fpexcept.strict">
; CHECK: %{{.*}} = llvm.call_intrinsic "llvm.experimental.constrained.ceil.f32"(%{{.*}}, %[[EB]]) : (f32, !llvm.metadata) -> f32
define float @constrained_ceil(float %a) {
  %r = call float @llvm.experimental.constrained.ceil.f32(float %a, metadata !"fpexcept.strict")
  ret float %r
}

; // -----

declare float @llvm.experimental.constrained.floor.f32(float, metadata)

; CHECK-LABEL: llvm.func @constrained_floor
; CHECK: llvm.mlir.metadata_as_value #llvm.md_string<"fpexcept.strict">
; CHECK: llvm.call_intrinsic "llvm.experimental.constrained.floor.f32"
define float @constrained_floor(float %a) {
  %r = call float @llvm.experimental.constrained.floor.f32(float %a, metadata !"fpexcept.strict")
  ret float %r
}

; // -----

declare float @llvm.experimental.constrained.round.f32(float, metadata)

; CHECK-LABEL: llvm.func @constrained_round
; CHECK: llvm.mlir.metadata_as_value #llvm.md_string<"fpexcept.strict">
; CHECK: llvm.call_intrinsic "llvm.experimental.constrained.round.f32"
define float @constrained_round(float %a) {
  %r = call float @llvm.experimental.constrained.round.f32(float %a, metadata !"fpexcept.strict")
  ret float %r
}

; // -----

declare float @llvm.experimental.constrained.roundeven.f32(float, metadata)

; CHECK-LABEL: llvm.func @constrained_roundeven
; CHECK: llvm.mlir.metadata_as_value #llvm.md_string<"fpexcept.strict">
; CHECK: llvm.call_intrinsic "llvm.experimental.constrained.roundeven.f32"
define float @constrained_roundeven(float %a) {
  %r = call float @llvm.experimental.constrained.roundeven.f32(float %a, metadata !"fpexcept.strict")
  ret float %r
}

; // -----

declare float @llvm.experimental.constrained.trunc.f32(float, metadata)

; CHECK-LABEL: llvm.func @constrained_trunc
; CHECK: llvm.mlir.metadata_as_value #llvm.md_string<"fpexcept.strict">
; CHECK: llvm.call_intrinsic "llvm.experimental.constrained.trunc.f32"
define float @constrained_trunc(float %a) {
  %r = call float @llvm.experimental.constrained.trunc.f32(float %a, metadata !"fpexcept.strict")
  ret float %r
}

; // -----

declare i32 @llvm.experimental.constrained.lround.i32.f32(float, metadata)

; CHECK-LABEL: llvm.func @constrained_lround
; CHECK: %[[EB:.*]] = llvm.mlir.metadata_as_value #llvm.md_string<"fpexcept.strict">
; CHECK: %{{.*}} = llvm.call_intrinsic "llvm.experimental.constrained.lround.i32.f32"(%{{.*}}, %[[EB]]) : (f32, !llvm.metadata) -> i32
define i32 @constrained_lround(float %a) {
  %r = call i32 @llvm.experimental.constrained.lround.i32.f32(float %a, metadata !"fpexcept.strict")
  ret i32 %r
}

; // -----

declare i64 @llvm.experimental.constrained.llround.i64.f32(float, metadata)

; CHECK-LABEL: llvm.func @constrained_llround
; CHECK: llvm.mlir.metadata_as_value #llvm.md_string<"fpexcept.strict">
; CHECK: llvm.call_intrinsic "llvm.experimental.constrained.llround.i64.f32"
define i64 @constrained_llround(float %a) {
  %r = call i64 @llvm.experimental.constrained.llround.i64.f32(float %a, metadata !"fpexcept.strict")
  ret i64 %r
}

; -----------------------------------------------------------------------------
; Constrained comparisons (fcmp / fcmps): predicate string + exception metadata.
; -----------------------------------------------------------------------------

; // -----

declare i1 @llvm.experimental.constrained.fcmp.f32(float, float, metadata, metadata)

; CHECK-LABEL: llvm.func @constrained_fcmp
; CHECK: %[[PRED:.*]] = llvm.mlir.metadata_as_value #llvm.md_string<"oeq">
; CHECK: %[[EB:.*]] = llvm.mlir.metadata_as_value #llvm.md_string<"fpexcept.strict">
; CHECK: %{{.*}} = llvm.call_intrinsic "llvm.experimental.constrained.fcmp.f32"(%{{.*}}, %{{.*}}, %[[PRED]], %[[EB]]) : (f32, f32, !llvm.metadata, !llvm.metadata) -> i1
define i1 @constrained_fcmp(float %a, float %b) {
  %r = call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %b, metadata !"oeq", metadata !"fpexcept.strict")
  ret i1 %r
}

; // -----

declare i1 @llvm.experimental.constrained.fcmps.f32(float, float, metadata, metadata)

; CHECK-LABEL: llvm.func @constrained_fcmps
; CHECK: %[[PRED:.*]] = llvm.mlir.metadata_as_value #llvm.md_string<"olt">
; CHECK: %[[EB:.*]] = llvm.mlir.metadata_as_value #llvm.md_string<"fpexcept.strict">
; CHECK: %{{.*}} = llvm.call_intrinsic "llvm.experimental.constrained.fcmps.f32"(%{{.*}}, %{{.*}}, %[[PRED]], %[[EB]]) : (f32, f32, !llvm.metadata, !llvm.metadata) -> i1
define i1 @constrained_fcmps(float %a, float %b) {
  %r = call i1 @llvm.experimental.constrained.fcmps.f32(float %a, float %b, metadata !"olt", metadata !"fpexcept.strict")
  ret i1 %r
}
