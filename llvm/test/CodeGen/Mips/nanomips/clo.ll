; RUN: llc -mtriple=nanomips -verify-machineinstrs < %s | FileCheck %s

define i32 @clo0(i32 %n) {
  %neg = xor i32 %n, -1
; CHECK: clo $a0, $a0
  %count = tail call i32 @llvm.ctlz.i32(i32 %neg, i1 true)
  ret i32 %count
}

define i32 @clo1(i32 %n) {
  %neg = xor i32 %n, -1
; CHECK: clo $a0, $a0
  %count = tail call i32 @llvm.ctlz.i32(i32 %neg, i1 true)
  ret i32 %count
}

declare i32 @llvm.ctlz.i32(i32, i1)
