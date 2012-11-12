; RUN: llc < %s -mtriple=x86_64-apple-darwin9 -mattr=sse42 | FileCheck %s -check-prefix=X64

declare i64 @llvm.x86.sse42.crc32.64.8(i64, i8) nounwind
declare i64 @llvm.x86.sse42.crc32.64.64(i64, i64) nounwind

define i64 @crc32_64_8(i64 %a, i8 %b) nounwind {
  %tmp = call i64 @llvm.x86.sse42.crc32.64.8(i64 %a, i8 %b)
  ret i64 %tmp

; X64: _crc32_64_8:
; X64:     crc32b   %sil,
}

define i64 @crc32_64_64(i64 %a, i64 %b) nounwind {
  %tmp = call i64 @llvm.x86.sse42.crc32.64.64(i64 %a, i64 %b)
  ret i64 %tmp

; X64: _crc32_64_64:
; X64:     crc32q   %rsi,
}

