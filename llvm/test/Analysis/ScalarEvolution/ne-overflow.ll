
; RUN: opt %s -passes='print<scalar-evolution>' -scalar-evolution-classify-expressions=0 2>&1 | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; A collection of tests focused on exercising logic to prove no-unsigned wrap
; from mustprogress semantics of loops.

; CHECK: Determining loop execution counts for: @test
; CHECK: Loop %for.body: Unpredictable backedge-taken count.
; CHECK: Determining loop execution counts for: @test_preinc
; CHECK: Loop %for.body: Unpredictable backedge-taken count.
; CHECK: Determining loop execution counts for: @test_well_defined_infinite_st
; CHECK: Loop %for.body: Unpredictable backedge-taken count.
; CHECK: Determining loop execution counts for: @test_well_defined_infinite_ld
; CHECK: Loop %for.body: Unpredictable backedge-taken count.
; CHECK: Determining loop execution counts for: @test_no_mustprogress
; CHECK: Loop %for.body: Unpredictable backedge-taken count.
; CHECK: Determining loop execution counts for: @test_1024
; CHECK: Loop %for.body: Unpredictable backedge-taken count.
; CHECK: Determining loop execution counts for: @test_uneven_divide
; CHECK: Loop %for.body: backedge-taken count is (-1 + (-1431655765 * %N))
; CHECK: Determining loop execution counts for: @test_non_invariant_rhs
; CHECK: Loop %for.body: Unpredictable backedge-taken count.
; CHECK: Determining loop execution counts for: @test_abnormal_exit
; CHECK: Loop %for.body: Unpredictable backedge-taken count.
; CHECK: Determining loop execution counts for: @test_other_exit
; CHECK: Loop %for.body: <multiple exits> Unpredictable backedge-taken count.

define void @test(i32 %N) mustprogress {
entry:
  br label %for.body

for.body:
  %iv = phi i32 [ %iv.next, %for.body ], [ 0, %entry ]
  %iv.next = add i32 %iv, 2
  %cmp = icmp ne i32 %iv.next, %N
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void
}

define void @test_preinc(i32 %N) mustprogress {
entry:
  br label %for.body

for.body:
  %iv = phi i32 [ %iv.next, %for.body ], [ 0, %entry ]
  %iv.next = add i32 %iv, 2
  %cmp = icmp ne i32 %iv, %N
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void

}

@G = external global i32

define void @test_well_defined_infinite_st(i32 %N) mustprogress {
entry:
  br label %for.body

for.body:
  %iv = phi i32 [ %iv.next, %for.body ], [ 0, %entry ]
  %iv.next = add i32 %iv, 2
  store volatile i32 0, i32* @G
  %cmp = icmp ne i32 %iv.next, %N
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void
}

define void @test_well_defined_infinite_ld(i32 %N) mustprogress {
entry:
  br label %for.body

for.body:
  %iv = phi i32 [ %iv.next, %for.body ], [ 0, %entry ]
  %iv.next = add i32 %iv, 2
  %val = load volatile i32, i32* @G
  %cmp = icmp ne i32 %iv.next, %N
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void
}

define void @test_no_mustprogress(i32 %N) {
entry:
  br label %for.body

for.body:
  %iv = phi i32 [ %iv.next, %for.body ], [ 0, %entry ]
  %iv.next = add i32 %iv, 2
  %cmp = icmp ne i32 %iv.next, %N
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void

}


define void @test_1024(i32 %N) mustprogress {
entry:
  br label %for.body

for.body:
  %iv = phi i32 [ %iv.next, %for.body ], [ 0, %entry ]
  %iv.next = add i32 %iv, 1024
  %cmp = icmp ne i32 %iv.next, %N
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void
}

define void @test_uneven_divide(i32 %N) mustprogress {
entry:
  br label %for.body

for.body:
  %iv = phi i32 [ %iv.next, %for.body ], [ 0, %entry ]
  %iv.next = add i32 %iv, 3
  %cmp = icmp ne i32 %iv.next, %N
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void
}

define void @test_non_invariant_rhs() mustprogress {
entry:
  br label %for.body

for.body:
  %iv = phi i32 [ %iv.next, %for.body ], [ 0, %entry ]
  %iv.next = add i32 %iv, 2
  %N = load i32, i32* @G
  %cmp = icmp ne i32 %iv.next, %N
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void
}

declare void @mayexit()

define void @test_abnormal_exit(i32 %N) mustprogress {
entry:
  br label %for.body

for.body:
  %iv = phi i32 [ %iv.next, %for.body ], [ 0, %entry ]
  %iv.next = add i32 %iv, 2
  call void @mayexit()
  %cmp = icmp ne i32 %iv.next, %N
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void
}


define void @test_other_exit(i32 %N) mustprogress {
entry:
  br label %for.body

for.body:
  %iv = phi i32 [ %iv.next, %for.latch ], [ 0, %entry ]
  %iv.next = add i32 %iv, 2
  %cmp1 = icmp ne i32 %iv.next, 20
  br i1 %cmp1, label %for.latch, label %for.cond.cleanup

for.latch:
  %cmp2 = icmp ne i32 %iv.next, %N
  br i1 %cmp2, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void
}


