; RUN: llc < %s -mtriple=nvptx64 -O3 -debug-only=dagcombine 2>&1 | FileCheck %s
; REQUIRES: asserts

; Test that combineMulWide creates the constant with the correct type (FromVT).
; This catches a bug where ToVT was incorrectly used instead of FromVT,
; resulting in a type mismatch (e.g., i32 constant instead of i16).

; CHECK: i32 = NVPTXISD::MUL_WIDE_SIGNED {{.*}}, Constant:i16<16384>
define i32 @shl_sext_i16(i16 %a) {
  %shl = shl nsw i16 %a, 14
  %conv = sext i16 %shl to i32
  ret i32 %conv
}

; CHECK: i32 = NVPTXISD::MUL_WIDE_UNSIGNED {{.*}}, Constant:i16<16384>
define i32 @shl_zext_i16(i16 %a) {
  %shl = shl nuw i16 %a, 14
  %conv = zext i16 %shl to i32
  ret i32 %conv
}

; CHECK: i64 = NVPTXISD::MUL_WIDE_SIGNED {{.*}}, Constant:i32<1073741824>
define i64 @shl_sext_i32(i32 %a) {
  %shl = shl nsw i32 %a, 30
  %conv = sext i32 %shl to i64
  ret i64 %conv
}

; CHECK: i64 = NVPTXISD::MUL_WIDE_UNSIGNED {{.*}}, Constant:i32<1073741824>
define i64 @shl_zext_i32(i32 %a) {
  %shl = shl nuw i32 %a, 30
  %conv = zext i32 %shl to i64
  ret i64 %conv
}

