; RUN: opt -loop-reduce -verify < %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64--linux-gnu"

define void @unbounded() {

block_A:
  %0 = sext i32 undef to i64
  br i1 undef, label %block_F, label %block_G

block_C:                ; preds = %block_F
  br i1 undef, label %block_D, label %block_E

block_D:                  ; preds = %block_D, %block_C
  br i1 undef, label %block_E, label %block_D

block_E:              ; preds = %block_D, %block_C
  %iv2 = phi i64 [ %4, %block_D ], [ %4, %block_C ]
  %1 = add nsw i32 %iv1, 1
  %2 = icmp eq i32 %1, undef
  br i1 %2, label %block_G, label %block_F

block_F:          ; preds = %block_E, %block_A
  %iv3 = phi i64 [ %iv2, %block_E ], [ %0, %block_A ]
  %iv1 = phi i32 [ %1, %block_E ], [ undef, %block_A ]
  %3 = add nsw i64 %iv3, 2
  %4 = add nsw i64 %iv3, 1
  br label %block_C

block_G:                              ; preds = %block_E, %block_A
  ret void
}
