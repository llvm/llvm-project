; RUN: opt -analyze -scalar-evolution %s | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

define void @test_and(i16 %in) {
  br label %bb2

bb2:                                              ; preds = %bb1.i, %bb2, %0
  %_tmp44.i = icmp slt i16 %in, 2
  br i1 %_tmp44.i, label %bb1.i, label %bb2

bb1.i:                                            ; preds = %bb1.i, %bb2
  %_tmp25.i = phi i16 [ %in, %bb2 ], [ %_tmp6.i, %bb1.i ]
  %_tmp6.i = add nsw i16 %_tmp25.i, 1
  %_tmp10.i = icmp sge i16 %_tmp6.i, 2
  %exitcond.i = icmp eq i16 %_tmp6.i, 2
  %or.cond = and i1 %_tmp10.i, %exitcond.i
  br i1 %or.cond, label %bb2, label %bb1.i
}

; CHECK-LABEL: Determining loop execution counts for: @test_and
; CHECK-NEXT: Loop %bb1.i: backedge-taken count is (1 + (-1 * %in))
; CHECK-NEXT: Loop %bb1.i: max backedge-taken count is -1
; CHECK-NEXT: Loop %bb1.i: Predicated backedge-taken count is (1 + (-1 * %in))


define void @test_or() {
  %C10 = icmp slt i1 undef, undef
  br i1 %C10, label %BB, label %exit

BB:                                               ; preds = %BB, %0
  %indvars.iv = phi i64 [ -1, %BB ], [ -1, %0 ]
  %sum.01 = phi i32 [ %2, %BB ], [ undef, %0 ]
  %1 = trunc i64 %indvars.iv to i32
  %2 = add nsw i32 %1, %sum.01
  %B3 = add i32 %1, %2
  %C11 = icmp ult i32 %2, %1
  %C5 = icmp sle i32 %1, %B3
  %B = or i1 %C5, %C11
  br i1 %B, label %BB, label %exit

exit:                                      ; preds = %BB, %0
  ret void
}

; CHECK-LABEL: Determining loop execution counts for: @test_or
; CHECK-NEXT: Loop %BB: backedge-taken count is undef
; CHECK-NEXT: Loop %BB: max backedge-taken count is -1
; CHECK-NEXT: Loop %BB: Predicated backedge-taken count is undef
