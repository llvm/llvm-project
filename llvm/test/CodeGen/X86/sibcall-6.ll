; RUN: llc < %s -relocation-model=pic | FileCheck %s
; PR15250

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32-S128"
target triple = "i386-unknown-linux-gnu"

declare void @callee1(i32 inreg, i32 inreg, i32 inreg)
define void @test1(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: test1:
; CHECK: calll callee1@PLT
  tail call void @callee1(i32 inreg 0, i32 inreg 0, i32 inreg 0) nounwind
  ret void
}
