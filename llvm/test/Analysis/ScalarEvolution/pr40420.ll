; RUN: opt < %s -disable-output "-passes=print<scalar-evolution>" 2>&1 2>&1 | FileCheck %s
; REQUIRES: asserts

define void @test(i8 %tmp6) {

; CHECK-LABEL: Classifying expressions for: @test

bb:
  br label %outer_loop

outer_loop:                                              ; preds = %outer_latch, %bb
  %tmp8 = phi i8 [ %tmp6, %bb ], [ %tmp20.lcssa, %outer_latch ]
  %tmp9 = phi i64 [ 0, %bb ], [ %tmp11, %outer_latch ]
  br label %inner_loop

outer_latch:                                             ; preds = %inner_loop
  %tmp20.lcssa = phi i8 [ %tmp20.6, %inner_loop ]
  %tmp11 = add i64 %tmp9, 239
  br label %outer_loop

inner_loop:                                             ; preds = %inner_latch, %outer_loop
  %tmp13 = phi i8 [ %tmp8, %outer_loop ], [ %tmp20.1, %inner_latch ]
  %tmp14 = phi i64 [ %tmp9, %outer_loop ], [ %tmp16.7, %inner_latch ]
  %tmp15 = phi i64 [ 2, %outer_loop ], [ %tmp21.7, %inner_latch ]
  %tmp16 = add i64 %tmp14, 1
  %tmp17 = trunc i64 %tmp16 to i8
  %tmp18 = mul i8 %tmp13, 100
  %tmp19 = mul i8 %tmp18, %tmp13
  %tmp20 = mul i8 %tmp19, %tmp17
  %tmp21 = add nuw nsw i64 %tmp15, 1
  %tmp16.1 = add i64 %tmp16, 1
  %tmp17.1 = trunc i64 %tmp16.1 to i8
  %tmp18.1 = mul i8 %tmp20, 100
  %tmp19.1 = mul i8 %tmp18.1, %tmp20
  %tmp20.1 = mul i8 %tmp19.1, %tmp17.1
  %tmp21.1 = add nuw nsw i64 %tmp21, 1
  %tmp16.2 = add i64 %tmp16.1, 1
  %tmp16.3 = add i64 %tmp16.2, 1
  %tmp16.4 = add i64 %tmp16.3, 1
  %tmp16.5 = add i64 %tmp16.4, 1
  %tmp16.6 = add i64 %tmp16.5, 1
  %tmp20.6 = mul i8 %tmp19.1, %tmp17.1
  %tmp22.6 = icmp ugt i64 %tmp16.1, 239
  br i1 %tmp22.6, label %outer_latch, label %inner_latch

inner_latch:                                           ; preds = %inner_loop
  %tmp16.7 = add i64 %tmp16.6, 1
  %tmp21.7 = add nuw nsw i64 %tmp21.1, 1
  br label %inner_loop
}
