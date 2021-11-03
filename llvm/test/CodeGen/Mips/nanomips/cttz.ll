; RUN: llc -mtriple=nanomips -verify-machineinstrs < %s | FileCheck %s

define i8 @cttz8(i8 %n) {
; CHECK: ori $a0, $a0, 256
; CHECK: bitrevw $a0, $a0
; CHECK: clz $a0, $a0
  %count = call i8 @llvm.cttz.i8(i8 %n)
  ret i8 %count
}

define i16 @cttz16(i16 %n) {
; CHECK: li $a1, 65536
; CHECK: or $a0, $a0, $a1
; CHECK: bitrevw $a0, $a0
; CHECK: clz $a0, $a0
  %count = call i16 @llvm.cttz.i16(i16 %n)
  ret i16 %count
}

define i32 @cttz32(i32 %n) {
; CHECK: bitrevw $a0, $a0
; CHECK: clz $a0, $a0
  %count = call i32 @llvm.cttz.i32(i32 %n)
  ret i32 %count
}

declare i8 @llvm.cttz.i8(i8)
declare i16 @llvm.cttz.i16(i16)
declare i32 @llvm.cttz.i32(i32)
