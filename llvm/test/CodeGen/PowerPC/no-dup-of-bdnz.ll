; RUN: opt -passes='early-cse<memssa>,loop-mssa(loop-rotate,licm,loop-rotate)' -earlycse-debug-hash -S %s -o - | FileCheck %s
; ModuleID = 'bugpoint-reduced-simplified.bc'
source_filename = "bugpoint-output-8903f29.bc"
target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

define void @test(i64 %arg.ssa, i64 %arg.nb) local_unnamed_addr {
; Ensure that loop rotation doesn't duplicate the call to
; llvm.loop.decrement
; CHECK-LABEL: test
; CHECK: call i1 @llvm.loop.decrement
; CHECK-NOT: call i1 @llvm.loop.decrement
; CHECK: declare i1 @llvm.loop.decrement
entry:
  switch i32 undef, label %BB_8 [
    i32 -2, label %BB_9
    i32 0, label %BB_9
  ]

BB_1:                                    ; preds = %BB_12, %BB_4
  %bcount.1.us = phi i64 [ %.810.us, %BB_4 ], [ 0, %BB_12 ]
  %0 = add i64 %arg.ssa, %bcount.1.us
  %.568.us = load i32, ptr undef, align 4
  %.15.i.us = icmp slt i32 0, %.568.us
  br i1 %.15.i.us, label %BB_3, label %BB_2

BB_2:                                          ; preds = %BB_1
  %.982.us = add nsw i64 %0, 1
  unreachable

BB_3:                                          ; preds = %BB_1
  %1 = add i64 %arg.ssa, %bcount.1.us
  %2 = add i64 %1, 1
  %3 = call i1 @llvm.loop.decrement.i32(i32 1)
  br i1 %3, label %BB_4, label %BB_7

BB_4:                                          ; preds = %BB_3
  %.810.us = add nuw nsw i64 %bcount.1.us, 1
  br label %BB_1

BB_5:                                         ; preds = %BB_7, %BB_5
  %lsr.iv20.i116 = phi i64 [ %2, %BB_7 ], [ %lsr.iv.next21.i126, %BB_5 ]
  %lsr.iv.next21.i126 = add i64 %lsr.iv20.i116, 1
  br i1 undef, label %BB_5, label %BB_6

BB_6:                                         ; preds = %BB_5
  ret void

BB_7:                                     ; preds = %BB_3
  br label %BB_5

BB_8:                                           ; preds = %entry
  ret void

BB_9:                                        ; preds = %entry, %entry
  br label %BB_10

BB_10:                               ; preds = %BB_9
  br label %BB_11

BB_11:                                         ; preds = %BB_11, %BB_10
  br i1 undef, label %BB_11, label %BB_12

BB_12:                                         ; preds = %BB_11
  call void @llvm.set.loop.iterations.i64(i64 %arg.nb)
  br label %BB_1
}

; Function Attrs: nounwind
declare void @llvm.set.loop.iterations.i64(i64) #0

; Function Attrs: nounwind
declare i1 @llvm.loop.decrement.i32(i32) #0

attributes #0 = { nounwind }
