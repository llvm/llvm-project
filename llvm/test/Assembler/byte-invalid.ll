; RUN: split-file %s %t
; RUN: not llvm-as -disable-output %t/zext-byte-to-int.ll   2>&1 | FileCheck %t/zext-byte-to-int.ll
; RUN: not llvm-as -disable-output %t/sext-byte-to-int.ll   2>&1 | FileCheck %t/sext-byte-to-int.ll
; RUN: not llvm-as -disable-output %t/trunc-byte-to-byte.ll 2>&1 | FileCheck %t/trunc-byte-to-byte.ll
; RUN: not llvm-as -disable-output %t/zext-int-to-byte.ll   2>&1 | FileCheck %t/zext-int-to-byte.ll
; RUN: not llvm-as -disable-output %t/sext-int-to-byte.ll   2>&1 | FileCheck %t/sext-int-to-byte.ll
; RUN: not llvm-as -disable-output %t/trunc-int-to-byte.ll  2>&1 | FileCheck %t/trunc-int-to-byte.ll
; RUN: not llvm-as -disable-output %t/trunc-byte-to-int.ll  2>&1 | FileCheck %t/trunc-byte-to-int.ll
; RUN: not llvm-as -disable-output %t/lshr-byte.ll          2>&1 | FileCheck %t/lshr-byte.ll
; RUN: not llvm-as -disable-output %t/icmp-byte.ll          2>&1 | FileCheck %t/icmp-byte.ll

;--- zext-byte-to-int.ll
; CHECK: invalid cast opcode for cast from 'b8' to 'i32'
define void @test(b8 %b) {
  %t = zext b8 %b to i32
  ret void
}

;--- sext-byte-to-int.ll
; CHECK: invalid cast opcode for cast from 'b8' to 'i32'
define void @test(b8 %b) {
  %t = sext b8 %b to i32
  ret void
}

;--- trunc-byte-to-byte.ll
; CHECK: invalid cast opcode for cast from 'b32' to 'b8'
define void @test(b32 %b) {
  %t = trunc b32 %b to b8
  ret void
}

;--- zext-int-to-byte.ll
; CHECK: invalid cast opcode for cast from 'i8' to 'b32'
define void @test(i8 %v) {
  %t = zext i8 %v to b32
  ret void
}

;--- sext-int-to-byte.ll
; CHECK: invalid cast opcode for cast from 'i8' to 'b32'
define void @test(i8 %v) {
  %t = sext i8 %v to b32
  ret void
}

;--- trunc-int-to-byte.ll
; CHECK: invalid cast opcode for cast from 'i32' to 'b8'
define void @test(i32 %v) {
  %t = trunc i32 %v to b8
  ret void
}

;--- trunc-byte-to-int.ll
; CHECK: invalid cast opcode for cast from 'b32' to 'i8'
define void @test(b32 %b) {
  %t = trunc b32 %b to i8
  ret void
}

;--- lshr-byte.ll
; CHECK: invalid operand type for instruction
define void @test(b32 %b) {
  %t = lshr b32 %b, 8
  ret void
}

;--- icmp-byte.ll
; CHECK: icmp requires integer operands
define void @test(b8 %b1, b8 %b2) {
  %cmp = icmp eq b8 %b1, %b2
  ret void
}
