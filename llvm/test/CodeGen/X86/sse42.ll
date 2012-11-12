; RUN: llc < %s -mtriple=i686-apple-darwin9 -mattr=sse42 | FileCheck %s -check-prefix=X32
; RUN: llc < %s -mtriple=x86_64-apple-darwin9 -mattr=sse42 | FileCheck %s -check-prefix=X64

declare i32 @llvm.x86.sse42.crc32.32.8(i32, i8) nounwind
declare i32 @llvm.x86.sse42.crc32.32.16(i32, i16) nounwind
declare i32 @llvm.x86.sse42.crc32.32.32(i32, i32) nounwind

define i32 @crc32_32_8(i32 %a, i8 %b) nounwind {
  %tmp = call i32 @llvm.x86.sse42.crc32.32.8(i32 %a, i8 %b)
  ret i32 %tmp
; X32: _crc32_32_8:
; X32:     crc32b   8(%esp), %eax

; X64: _crc32_32_8:
; X64:     crc32b   %sil,
}


define i32 @crc32_32_16(i32 %a, i16 %b) nounwind {
  %tmp = call i32 @llvm.x86.sse42.crc32.32.16(i32 %a, i16 %b)
  ret i32 %tmp
; X32: _crc32_32_16:
; X32:     crc32w   8(%esp), %eax

; X64: _crc32_32_16:
; X64:     crc32w   %si,
}


define i32 @crc32_32_32(i32 %a, i32 %b) nounwind {
  %tmp = call i32 @llvm.x86.sse42.crc32.32.32(i32 %a, i32 %b)
  ret i32 %tmp
; X32: _crc32_32_32:
; X32:     crc32l   8(%esp), %eax

; X64: _crc32_32_32:
; X64:     crc32l   %esi,
}

