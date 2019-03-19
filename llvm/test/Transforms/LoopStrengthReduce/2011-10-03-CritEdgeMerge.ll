; RUN: opt -loop-reduce -S < %s | FileCheck %s
;
; Test LSR's use of SplitCriticalEdge during phi rewriting.

target triple = "x86_64-apple-darwin"

; Provide legal integer types.
target datalayout = "n8:16:32:64"


; Verify that identical edges are merged. rdar://problem/6453893
; CHECK-LABEL: @test1(
; CHECK: bb89:
; CHECK: phi i8* [ %lsr.iv.next1, %bbA.bb89_crit_edge ], [ %lsr.iv.next1, %bbB.bb89_crit_edge ]{{$}}

define i8* @test1() {
entry:
  br label %loop

loop:
  %rec = phi i32 [ %next, %loop ], [ 0, %entry ]
  %next = add i32 %rec, 1
  %tmp75 = getelementptr i8, i8* null, i32 %next
  br i1 false, label %loop, label %loopexit

loopexit:
  br i1 false, label %bbA, label %bbB

bbA:
  switch i32 0, label %bb89 [
    i32 47, label %bb89
    i32 58, label %bb89
  ]

bbB:
  switch i8 0, label %bb89 [
    i8 47, label %bb89
    i8 58, label %bb89
  ]

bb89:
  %tmp75phi = phi i8* [ %tmp75, %bbA ], [ %tmp75, %bbA ], [ %tmp75, %bbA ], [ %tmp75, %bbB ], [ %tmp75, %bbB ], [ %tmp75, %bbB ]
  br label %exit

exit:
  ret i8* %tmp75phi
}

; Handle single-predecessor phis: PR13756
; CHECK-LABEL: @test2(
; CHECK: bb89:
; CHECK: phi i8* [ %lsr.iv.next1, %bbA ], [ %lsr.iv.next1, %bbA ], [ %lsr.iv.next1, %bbA ]{{$}}
define i8* @test2() {
entry:
  br label %loop

loop:
  %rec = phi i32 [ %next, %loop ], [ 0, %entry ]
  %next = add i32 %rec, 1
  %tmp75 = getelementptr i8, i8* null, i32 %next
  br i1 false, label %loop, label %loopexit

loopexit:
  br i1 false, label %bbA, label %bbB

bbA:
  switch i32 0, label %bb89 [
    i32 47, label %bb89
    i32 58, label %bb89
  ]

bbB:
  switch i8 0, label %exit [
    i8 47, label %exit
    i8 58, label %exit
  ]

bb89:
  %tmp75phi = phi i8* [ %tmp75, %bbA ], [ %tmp75, %bbA ], [ %tmp75, %bbA ]
  br label %exit

exit:
  %result = phi i8* [ %tmp75phi, %bb89 ], [ %tmp75, %bbB ], [ %tmp75, %bbB ], [ %tmp75, %bbB ]
  ret i8* %result
}
