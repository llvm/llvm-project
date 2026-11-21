;; Test verification of arbitrary FP conversion intrinsics:
;; - Metadata validation (interpretation, rounding mode)
;; - Bit-width validation
; RUN: split-file %s %t
; RUN: not llvm-as %t/bad-interpretation-empty.ll -disable-output 2>&1 | FileCheck %s --check-prefix=BAD-INTERP-EMPTY
; RUN: not llvm-as %t/bad-interpretation-unknown.ll -disable-output 2>&1 | FileCheck %s --check-prefix=BAD-INTERP-UNKNOWN
; RUN: not llvm-as %t/bad-rounding.ll -disable-output 2>&1 | FileCheck %s --check-prefix=BAD-ROUNDING
; RUN: not llvm-as %t/width-to-mismatch.ll -disable-output 2>&1 | FileCheck %s --check-prefix=WIDTH-TO-MISMATCH
; RUN: not llvm-as %t/width-from-mismatch.ll -disable-output 2>&1 | FileCheck %s --check-prefix=WIDTH-FROM-MISMATCH
; RUN: not llvm-as %t/width-vec-mismatch.ll -disable-output 2>&1 | FileCheck %s --check-prefix=WIDTH-VEC-MISMATCH

;--- bad-interpretation-empty.ll
; BAD-INTERP-EMPTY: interpretation metadata string must not be empty

declare i8 @llvm.convert.to.arbitrary.fp.i8.f16(half, metadata, metadata, i1)

define i8 @bad_interpretation_empty(half %v) {
  %r = call i8 @llvm.convert.to.arbitrary.fp.i8.f16(
      half %v, metadata !"", metadata !"round.tonearest", i1 false)
  ret i8 %r
}

;--- bad-interpretation-unknown.ll
; BAD-INTERP-UNKNOWN: unsupported interpretation metadata string

declare i8 @llvm.convert.to.arbitrary.fp.i8.f16(half, metadata, metadata, i1)

define i8 @bad_interpretation_unknown(half %v) {
  %r = call i8 @llvm.convert.to.arbitrary.fp.i8.f16(
      half %v, metadata !"unknown", metadata !"round.tonearest", i1 false)
  ret i8 %r
}

;--- bad-rounding.ll
; BAD-ROUNDING: unsupported rounding mode argument

declare i8 @llvm.convert.to.arbitrary.fp.i8.f16(half, metadata, metadata, i1)

define i8 @bad_rounding(half %v) {
  %r = call i8 @llvm.convert.to.arbitrary.fp.i8.f16(
      half %v, metadata !"Float8E4M3", metadata !"round.dynamic", i1 false)
  ret i8 %r
}

;--- width-to-mismatch.ll
; WIDTH-TO-MISMATCH: integer type bit width must equal the arbitrary FP format width

declare i8 @llvm.convert.to.arbitrary.fp.i8.f16(half, metadata, metadata, i1)

define i8 @bad_width_to(half %v) {
  %r = call i8 @llvm.convert.to.arbitrary.fp.i8.f16(
      half %v, metadata !"Float4E2M1FN", metadata !"round.tonearest", i1 false)
  ret i8 %r
}

;--- width-from-mismatch.ll
; WIDTH-FROM-MISMATCH: integer type bit width must equal the arbitrary FP format width

declare float @llvm.convert.from.arbitrary.fp.f32.i8(i8, metadata)

define float @bad_width_from(i8 %v) {
  %r = call float @llvm.convert.from.arbitrary.fp.f32.i8(
      i8 %v, metadata !"Float4E2M1FN")
  ret float %r
}

;--- width-vec-mismatch.ll
; WIDTH-VEC-MISMATCH: integer type bit width must equal the arbitrary FP format width

declare <4 x i8> @llvm.convert.to.arbitrary.fp.v4i8.v4f16(<4 x half>, metadata, metadata, i1)

define <4 x i8> @bad_width_vec(<4 x half> %v) {
  %r = call <4 x i8> @llvm.convert.to.arbitrary.fp.v4i8.v4f16(
      <4 x half> %v, metadata !"Float4E2M1FN", metadata !"round.tonearest", i1 false)
  ret <4 x i8> %r
}
