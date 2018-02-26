; RUN: llc %s -o - | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @test1() #0 {
entry:
  ret void

; CHECK-LABEL: @test1
; CHECK: callq __fentry__
; CHECK-NOT: mcount
; CHECK: retq
}

define void @test2() #1 {
entry:
  br label %bb1
bb1:
  call void @address_taken(i64 ptrtoint (i8* blockaddress(@test2, %bb1) to i64), i32 512)
  ret void

; CHECK-LABEL: @test2
; CHECK: callq __fentry__
; CHECK-NOT: mcount
; CHECK: retq
}

declare void @address_taken(i64, i32) local_unnamed_addr
attributes #0 = { "fentry-call"="true" }
attributes #1 = { inlinehint minsize noredzone nounwind optsize sspstrong "fentry-call"="true" }
