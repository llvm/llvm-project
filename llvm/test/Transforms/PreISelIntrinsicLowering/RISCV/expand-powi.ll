; RUN: opt -passes=pre-isel-intrinsic-lowering -mtriple=riscv64 -S < %s | FileCheck %s

; The exponent is scalar and must not be extracted from the vector.
define <vscale x 2 x double> @powi(<vscale x 2 x double> %x, i32 %n) {
; CHECK-LABEL: define <vscale x 2 x double> @powi(
; CHECK:       [[LOOP:.*]]:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64
; CHECK-NEXT:    [[VECTOR:%.*]] = phi <vscale x 2 x double>
; CHECK-NEXT:    [[BASE:%.*]] = extractelement <vscale x 2 x double> %x, i64 [[INDEX]]
; CHECK-NEXT:    [[RESULT:%.*]] = call double @llvm.powi.f64.i32(double [[BASE]], i32 %n)
; CHECK-NEXT:    [[INSERT:%.*]] = insertelement <vscale x 2 x double> [[VECTOR]], double [[RESULT]], i64 [[INDEX]]
;
  %result = call <vscale x 2 x double> @llvm.powi.nxv2f64.i32(<vscale x 2 x double> %x, i32 %n)
  ret <vscale x 2 x double> %result
}

; Preserve non-i32 exponent types and fast-math flags.
define <vscale x 2 x double> @powi_i64(<vscale x 2 x double> %x, i64 %n) {
; CHECK-LABEL: define <vscale x 2 x double> @powi_i64(
; CHECK:       call fast double @llvm.powi.f64.i64(double {{%.*}}, i64 %n)
;
  %result = call fast <vscale x 2 x double> @llvm.powi.nxv2f64.i64(<vscale x 2 x double> %x, i64 %n)
  ret <vscale x 2 x double> %result
}

; Leave constant exponents for SelectionDAGBuilder::ExpandPowI.
define <vscale x 2 x double> @powi_constant(<vscale x 2 x double> %x) {
; CHECK-LABEL: define <vscale x 2 x double> @powi_constant(
; CHECK-NEXT:    [[RESULT:%.*]] = call <vscale x 2 x double> @llvm.powi.nxv2f64.i32(<vscale x 2 x double> %x, i32 100)
; CHECK-NEXT:    ret <vscale x 2 x double> [[RESULT]]
;
  %result = call <vscale x 2 x double> @llvm.powi.nxv2f64.i32(<vscale x 2 x double> %x, i32 100)
  ret <vscale x 2 x double> %result
}

; Keep the constrained intrinsic and its metadata operands intact.
define <vscale x 2 x double> @constrained_powi(<vscale x 2 x double> %x, i32 %n) {
; CHECK-LABEL: define <vscale x 2 x double> @constrained_powi(
; CHECK:       call double @llvm.experimental.constrained.powi.f64(double {{%.*}}, i32 %n, metadata !"round.dynamic", metadata !"fpexcept.strict")

;
  %result = call <vscale x 2 x double> @llvm.experimental.constrained.powi.nxv2f64.i32(<vscale x 2 x double> %x, i32 %n, metadata !"round.dynamic", metadata !"fpexcept.strict") strictfp
  ret <vscale x 2 x double> %result
}
