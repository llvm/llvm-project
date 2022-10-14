; RUN: opt < %s -passes=indvars -S | FileCheck %s

; This is regression test for the bug in ScalarEvolution::isKnownPredicate.
; It does not check whether SCEV is available at loop entry before invoking
; and utility function isLoopEntryGuardedByCond and that leads to miscompile.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128-ni:1"
target triple = "x86_64-unknown-linux-gnu"

declare void @foo(i64)
declare void @bar(i32)

define void @test(i8* %arr) {
entry:
  br label %outer_header

outer_header:
  %i = phi i32 [40, %entry], [%i.next, %outer_latch]
  %i.64 = sext i32 %i to i64
  br label %inner_header

inner_header:
  %j = phi i32 [27, %outer_header], [%j.next, %inner_backedge]
  %j1 = zext i32 %j to i64
; The next 4 lines are required for avoid widening of %j and
; SCEV at %cmp would not be AddRec.
  %gep = getelementptr inbounds i8, i8*  %arr, i64 %j1
  %ld = load i8, i8* %gep
  %ec = icmp eq i8 %ld, 0
  br i1 %ec, label %return, label %inner_backedge

inner_backedge:
  %cmp = icmp ult i32 %j, %i
  %s = select i1 %cmp, i32 %i, i32 %j
; Select should not be simplified because if
; %i == 26 and %j == 27, %s should be equal to %j.
; In case of a bug the instruction is simplified to
; %s = select i1 true, i32 %0, i32 %j
; CHECK-NOT: %s = select i1 true
  call void @bar(i32 %s)
  %j.next = add nsw i32 %j, -2
  %cond = icmp ult i32 %j, 3
  br i1 %cond, label %outer_latch, label %inner_header

outer_latch:
  %i.next = add i32 %i, -1
  %cond2 = icmp sgt i32 %i.next, 13
; This line is just for forcing widening of %i
  call void @foo(i64 %i.64)
  br i1 %cond2, label %outer_header, label %return

return:
  ret void
}
