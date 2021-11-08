; RUN: llc -mtriple=nanomips -verify-machineinstrs < %s | FileCheck %s

define signext i16 @bswap_i16(i16 signext %n) {
; CHECK: byterevw $a0, $a0
; CHECK: sra $a0, $a0, 16
  %swapped = call i16 @llvm.bswap.i16(i16 %n)
  ret i16 %swapped
}

define zeroext i16 @bswap_u16(i16 zeroext %n) {
; CHECK: byterevw $a0, $a0
; CHECK: srl $a0, $a0, 16
  %swapped = call i16 @llvm.bswap.i16(i16 %n)
  ret i16 %swapped
}

define i32 @bswap_i32(i32 signext %n) {
; CHECK: byterevw $a0, $a0
  %swapped = call i32 @llvm.bswap.i32(i32 %n)
  ret i32 %swapped
}

define i32 @bswap_u32(i32 zeroext %n) {
; CHECK: byterevw $a0, $a0
  %swapped = call i32 @llvm.bswap.i32(i32 %n)
  ret i32 %swapped
}

define i64 @bswap_i64(i64 signext %n) {
; CHECK: byterevw $a2, $a1
; CHECK: byterevw $a1, $a0
; CHECK: move $a0, $a2
  %swapped = call i64 @llvm.bswap.i64(i64 %n)
  ret i64 %swapped
}

define i64 @bswap_u64(i64 zeroext %n) {
; CHECK: byterevw $a2, $a1
; CHECK: byterevw $a1, $a0
; CHECK: move $a0, $a2
  %swapped = call i64 @llvm.bswap.i64(i64 %n)
  ret i64 %swapped
}

declare i64 @llvm.bswap.i64(i64)
declare i32 @llvm.bswap.i32(i32)
declare i16 @llvm.bswap.i16(i16)
