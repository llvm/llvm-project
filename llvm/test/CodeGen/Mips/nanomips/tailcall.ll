; RUN: llc -mtriple=nanomips -asm-show-inst -verify-machineinstrs < %s | FileCheck %s

define internal i32 @g(i32 %x) {
  %add = add nsw i32 %x, 1
  ret i32 %add
}

define i32 @f(i32 %x) {
; CHECK: bc g
  %call = tail call i32 @g(i32 %x)
  ret i32 %call
}
