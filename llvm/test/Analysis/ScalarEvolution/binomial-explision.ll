; RUN: opt -analyze -scalar-evolution < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128-ni:1"

; Check that we don't have unreasonably huge SCEVs and in particular only a
; reasonable amount of AddRecs in the notation of %tmp19. If we "simplify" SCEVs
; too aggressively, we may end up with huge nested expressions.
define void @test(i32 %x, i64 %y, i1 %cond) {

; CHECK: %tmp19 = mul i32 %tmp17, %tmp18
; CHECK: ((((((
; CHECK-NOT: (((((
; CHECK: %tmp20 = add i32 %tmp19, %x

bb:
  br label %bb1

bb1:                                              ; preds = %bb3, %bb
  %tmp = phi i64 [ %y, %bb ], [ %tmp22, %bb3 ]
  %tmp2 = phi i32 [ %x, %bb ], [ %tmp4, %bb3 ]
  br label %bb5

bb3:                                              ; preds = %bb5
  %tmp4 = add i32 %tmp2, %x
  br label %bb1

bb5:                                              ; preds = %bb5, %bb1
  %tmp6 = phi i32 [ %tmp23, %bb5 ], [ %tmp2, %bb1 ]
  %tmp7 = sub i32 -119, %tmp6
  %tmp8 = mul i32 %tmp7, %x
  %tmp9 = sub i32 -120, %tmp6
  %tmp10 = mul i32 %tmp8, %tmp9
  %tmp11 = mul i32 %x, %tmp10
  %tmp12 = sub i32 -121, %tmp6
  %tmp13 = mul i32 %tmp10, %tmp12
  %tmp14 = mul i32 %tmp11, %tmp13
  %tmp15 = sub i32 -122, %tmp6
  %tmp16 = mul i32 %tmp13, %tmp15
  %tmp17 = mul i32 %tmp14, %tmp16
  %tmp18 = mul i32 %tmp16, %x
  %tmp19 = mul i32 %tmp17, %tmp18
  %tmp20 = add i32 %tmp19, %x
  %tmp21 = sext i32 %tmp20 to i64
  %tmp22 = add i64 %y, %tmp21
  %tmp23 = add i32 %tmp6, 7
  br i1 %cond, label %bb5, label %bb3
}
