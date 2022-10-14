; RUN: llc %s -mtriple=s390x-linux-gnu -mcpu=z10 -o - -verify-machineinstrs \
; RUN:   | FileCheck %s

define void @test1() #0 {
entry:
  ret void

; CHECK-LABEL: @test1
; CHECK: brasl %r0, __fentry__
; CHECK-NOT: mcount
; CHECK: br %r14
}

define void @test2() #1 {
entry:
  br label %bb1
bb1:
  call void @address_taken(i64 ptrtoint (ptr blockaddress(@test2, %bb1) to i64), i32 512)
  ret void

; CHECK-LABEL: @test2
; CHECK: brasl %r0, __fentry__
; CHECK-NOT: mcount
; CHECK: br %r14
}

declare void @address_taken(i64, i32) local_unnamed_addr
attributes #0 = { "fentry-call"="true" }
attributes #1 = { inlinehint minsize noredzone nounwind optsize sspstrong "fentry-call"="true" }
