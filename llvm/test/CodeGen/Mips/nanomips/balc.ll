; RUN: llc -mtriple=nanomips -asm-show-inst -verify-machineinstrs < %s | FileCheck %s

define i32 @foo(i32 %a, i32 %b) {
  %1 = add i32 %a, %b
  ret i32 %1
}

define i32 @bar(i32 %a, i32 %b) {
; CHECK: balc foo
; CHECK: BALC_NM
  %1 = call i32 @foo(i32 %a, i32 %b)
  ret i32 %1
}
