; RUN: llc < %s -debug-only=isel -O3 -mattr=avx  2>&1 | FileCheck %s

; Bug 45563:
; The LowerMLOAD() method AVX masked load branch should
; use the operand vector type rather than the mask type.
; Given, for example:
;   v4f64,ch = masked_load ..
; The select should be:
;   v4f64 = vselect ..
; instead of:
;   v4i64 = vselect ..

define <16 x double> @bug45563(<16 x double>* %addr, <16 x double> %dst, <16 x i64> %e, <16 x i64> %f) {
; CHECK-LABEL: bug45563:
; CHECK:       v4f64 = vselect
  %mask = icmp slt <16 x i64> %e, %f
  %res = call <16 x double> @llvm.masked.load.v16f64.p0v16f64(<16 x double>* %addr, i32 4, <16 x i1>%mask, <16 x double> %dst)
  ret <16 x double> %res
}

declare <16 x double> @llvm.masked.load.v16f64.p0v16f64(<16 x double>* %addr, i32 %align, <16 x i1> %mask, <16 x double> %dst)
