; RUN: llc -mtriple=nanomips -verify-machineinstrs < %s | FileCheck %s

define i32 @clz0(i32 %n) {
; CHECK: clz $a0, $a0
  %count = tail call i32 @llvm.ctlz.i32(i32 %n, i1 true)
  ret i32 %count
}

define i32 @clz1(i32 %n) {
; CHECK: clz $a0, $a0
  %count = tail call i32 @llvm.ctlz.i32(i32 %n, i1 true)
  ret i32 %count
}

declare i32 @llvm.ctlz.i32(i32, i1)
