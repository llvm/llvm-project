; RUN: llc -mtriple=nanomips -verify-machineinstrs < %s | FileCheck %s

define i32 @bitrev32(i32 %n) {
; CHECK: bitrevw $a0, $a0
  %rev = call i32 @llvm.bitreverse.i32(i32 %n)
  ret i32 %rev
}

define i16 @bitrev16(i16 %n) {
; CHECK: bitrevw $a0, $a0
; CHECK: srl $a0, $a0, 16
  %rev = call i16 @llvm.bitreverse.i16(i16 %n)
  ret i16 %rev
}

define i8 @bitrev8(i8 %n) {
; CHECK: bitrevw $a0, $a0
; CHECK: srl $a0, $a0, 24
  %rev = call i8 @llvm.bitreverse.i8(i8 %n)
  ret i8 %rev
}

declare i32 @llvm.bitreverse.i32(i32)
declare i16 @llvm.bitreverse.i16(i16)
declare i8 @llvm.bitreverse.i8(i8)
