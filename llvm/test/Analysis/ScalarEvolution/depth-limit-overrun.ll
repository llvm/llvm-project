; RUN: opt -passes 'loop-reduce' -scalar-evolution-max-arith-depth=2 -S < %s | FileCheck %s
; RUN: opt -loop-reduce -scalar-evolution-max-arith-depth=2 -S < %s | FileCheck %s

; This test should just compile cleanly without assertions.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:1-p2:32:8:8:32-ni:2"

define void @test(i32 %A, i32 %B, i32 %C) {
; CHECK-LABEL: @test(
; CHECK:       inner_loop:
; CHECK-NEXT:    [[LSR_IV3:%.*]] = phi i32
; CHECK-NEXT:    [[LSR_IV1:%.*]] = phi i32
; CHECK-NEXT:    [[LSR_IV:%.*]] = phi i32
; CHECK:         [[LSR_IV_NEXT:%.*]] = add i32 [[LSR_IV]], 3
; CHECK-NEXT:    [[LSR_IV_NEXT2:%.*]] = add i32 [[LSR_IV1]], 3
; CHECK-NEXT:    [[LSR_IV_NEXT4:%.*]] = add i32 [[LSR_IV3]], -3
;
entry:
  br label %outer_loop

outer_loop:
  %phi2 = phi i32 [ %A, %entry ], [ 204, %outer_tail ]
  %phi3 = phi i32 [ %A, %entry ], [ 243, %outer_tail ]
  %phi4 = phi i32 [ %B, %entry ], [ %i35, %outer_tail ]
  br label %guard

guard:
  %lcmp.mod = icmp eq i32 %C, 0
  br i1 %lcmp.mod, label %outer_tail, label %preheader

preheader:
  %i15 = shl i32 %B, 1
  br label %inner_loop

inner_loop:
  %phi5 = phi i32 [ %phi3, %preheader ], [ %i30, %inner_loop ]
  %phi6 = phi i32 [ %phi2, %preheader ], [ %i33, %inner_loop ]
  %iter = phi i32 [ %C, %preheader ], [ %iter.sub, %inner_loop ]
  %i17 = sub i32 %phi4, %phi6
  %i18 = sub i32 14, %phi5
  %i19 = mul i32 %i18, %C
  %factor.prol = shl i32 %phi5, 1
  %i20 = add i32 %i17, %factor.prol
  %i21 = add i32 %i20, %B
  %i22 = add i32 %i21, %i19
  %i23 = sub i32 14, %i22
  %i24 = mul i32 %i23, %C
  %factor.1.prol = shl i32 %i22, 1
  %i25 = add i32 %i17, %factor.1.prol
  %i27 = add i32 %i25, %i24
  %i29 = mul i32 %i25, %C
  %factor.2.prol = shl i32 %i27, 1
  %i30 = add i32 %i17, %factor.2.prol
  %i33 = add nsw i32 %phi6, -3
  %iter.sub = add i32 %iter, -1
  %iter.cmp = icmp eq i32 %iter.sub, 0
  br i1 %iter.cmp, label %outer_tail, label %inner_loop

outer_tail:
  %phi7 = phi i32 [ %phi2, %guard ], [ %i33, %inner_loop ]
  %i35 = sub i32 %A, %phi7
  %cmp = icmp sgt i32 %i35, 9876
  br i1 %cmp, label %exit, label %outer_loop

exit:
  ret void

}
