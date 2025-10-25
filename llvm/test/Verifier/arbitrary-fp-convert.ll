; RUN: split-file %s %t
; RUN: not opt -S -passes=verify %t/bad-result.ll 2>&1 | FileCheck %s --check-prefix=BADRESULT
; RUN: not opt -S -passes=verify %t/bad-rounding.ll 2>&1 | FileCheck %s --check-prefix=BADROUND
; RUN: not opt -S -passes=verify %t/bad-saturation.ll 2>&1 | FileCheck %s --check-prefix=BADSAT
; RUN: opt -S -passes=verify %t/good.ll

;--- bad-result.ll
; BADRESULT: result interpretation metadata string must not be empty
declare half @llvm.arbitrary.fp.convert.half.i8(i8, metadata, metadata, metadata, i32)

define half @bad_result(i8 %v) {
  %r = call half @llvm.arbitrary.fp.convert.half.i8(
      i8 %v, metadata !"", metadata !"spv.E5M2EXT", metadata !"none", i32 0)
  ret half %r
}

;--- bad-rounding.ll
; BADROUND: unsupported rounding mode argument
declare i8 @llvm.arbitrary.fp.convert.i8.half(half, metadata, metadata, metadata, i32)

define i8 @bad_rounding(half %v) {
  %r = call i8 @llvm.arbitrary.fp.convert.i8.half(
      half %v, metadata !"spv.E4M3EXT", metadata !"none", metadata !"round.dynamic", i32 0)
  ret i8 %r
}

;--- bad-saturation.ll
; BADSAT: saturation operand must be 0 or 1
declare i8 @llvm.arbitrary.fp.convert.i8.half.sat(half, metadata, metadata, metadata, i32)

define i8 @bad_saturation(half %v) {
  %r = call i8 @llvm.arbitrary.fp.convert.i8.half.sat(
      half %v, metadata !"spv.E4M3EXT", metadata !"none", metadata !"round.towardzero", i32 2)
  ret i8 %r
}

;--- good.ll
declare half @llvm.arbitrary.fp.convert.half.i8(i8, metadata, metadata, metadata, i32)
declare i8 @llvm.arbitrary.fp.convert.i8.half(half, metadata, metadata, metadata, i32)
declare i32 @llvm.arbitrary.fp.convert.i32.i8(i8, metadata, metadata, metadata, i32)
declare i8 @llvm.arbitrary.fp.convert.i8.i32(i32, metadata, metadata, metadata, i32)

define half @good_from(i8 %v) {
  %r = call half @llvm.arbitrary.fp.convert.half.i8(
      i8 %v, metadata !"none", metadata !"spv.E4M3EXT", metadata !"none", i32 0)
  ret half %r
}

define i8 @good_to(half %v) {
  %r = call i8 @llvm.arbitrary.fp.convert.i8.half(
      half %v, metadata !"spv.E4M3EXT", metadata !"none", metadata !"round.towardzero", i32 0)
  ret i8 %r
}

; Test integer conversions with rounding modes - these are now allowed
define i32 @good_int_rounding(i8 %v) {
  %r = call i32 @llvm.arbitrary.fp.convert.i32.i8(
      i8 %v, metadata !"signed", metadata !"spv.E4M3EXT", metadata !"none", i32 0)
  ret i32 %r
}

define i8 @good_input_rounding(i32 %v) {
  %r = call i8 @llvm.arbitrary.fp.convert.i8.i32(
      i32 %v, metadata !"spv.E4M3EXT", metadata !"signed", metadata !"none", i32 0)
  ret i8 %r
}
