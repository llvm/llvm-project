; RUN: opt -S -loop-reduce < %s | FileCheck %s

; Check that no crash here.
; When GenerateICmpZeroScales transforms the base formula
; it can get non-canonical form.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @hoge(i32 %arg) {
; CHECK: @hoge
bb:
  %tmp = and i32 %arg, -8
  br label %bb2

bb1:                                              ; preds = %bb2
  ret void

bb2:                                              ; preds = %bb2, %bb
  %tmp3 = phi i64 [ 0, %bb ], [ %tmp62, %bb2 ]
  %tmp4 = phi i32 [ 1, %bb ], [ %tmp63, %bb2 ]
  %tmp5 = phi i32 [ 0, %bb ], [ %tmp64, %bb2 ]
  %tmp6 = add i64 %tmp3, 1
  %tmp7 = trunc i64 %tmp6 to i32
  %tmp8 = sub i32 %tmp4, %tmp7
  %tmp9 = mul i32 %tmp8, %tmp8
  %tmp10 = sub i32 %tmp9, %tmp8
  %tmp11 = sext i32 %tmp10 to i64
  %tmp12 = sub i64 %tmp6, %tmp11
  %tmp13 = add nuw nsw i32 %tmp4, 1
  %tmp14 = add i64 %tmp12, 1
  %tmp15 = trunc i64 %tmp14 to i32
  %tmp16 = sub i32 %tmp13, %tmp15
  %tmp17 = mul i32 %tmp16, %tmp16
  %tmp18 = sub i32 %tmp17, %tmp16
  %tmp19 = sext i32 %tmp18 to i64
  %tmp20 = sub i64 %tmp14, %tmp19
  %tmp21 = add i64 %tmp20, 1
  %tmp22 = sub i64 %tmp21, 0
  %tmp23 = add nuw nsw i32 %tmp4, 3
  %tmp24 = add i64 %tmp22, 1
  %tmp25 = trunc i64 %tmp24 to i32
  %tmp26 = sub i32 %tmp23, %tmp25
  %tmp27 = mul i32 %tmp26, %tmp26
  %tmp28 = sub i32 %tmp27, %tmp26
  %tmp29 = sext i32 %tmp28 to i64
  %tmp30 = sub i64 %tmp24, %tmp29
  %tmp31 = add nuw nsw i32 %tmp4, 4
  %tmp32 = add i64 %tmp30, 1
  %tmp33 = trunc i64 %tmp32 to i32
  %tmp34 = sub i32 %tmp31, %tmp33
  %tmp35 = mul i32 %tmp34, %tmp34
  %tmp36 = sub i32 %tmp35, %tmp34
  %tmp37 = sext i32 %tmp36 to i64
  %tmp38 = sub i64 %tmp32, %tmp37
  %tmp39 = add nuw nsw i32 %tmp4, 5
  %tmp40 = add i64 %tmp38, 1
  %tmp41 = trunc i64 %tmp40 to i32
  %tmp42 = sub i32 %tmp39, %tmp41
  %tmp43 = mul i32 %tmp42, %tmp42
  %tmp44 = sub i32 %tmp43, %tmp42
  %tmp45 = sext i32 %tmp44 to i64
  %tmp46 = sub i64 %tmp40, %tmp45
  %tmp47 = add nuw nsw i32 %tmp4, 6
  %tmp48 = add i64 %tmp46, 1
  %tmp49 = trunc i64 %tmp48 to i32
  %tmp50 = sub i32 %tmp47, %tmp49
  %tmp51 = mul i32 %tmp50, %tmp50
  %tmp52 = sub i32 %tmp51, %tmp50
  %tmp53 = sext i32 %tmp52 to i64
  %tmp54 = sub i64 %tmp48, %tmp53
  %tmp55 = add nuw nsw i32 %tmp4, 7
  %tmp56 = add i64 %tmp54, 1
  %tmp57 = trunc i64 %tmp56 to i32
  %tmp58 = sub i32 %tmp55, %tmp57
  %tmp59 = mul i32 %tmp58, %tmp58
  %tmp60 = sub i32 %tmp59, %tmp58
  %tmp61 = sext i32 %tmp60 to i64
  %tmp62 = sub i64 %tmp56, %tmp61
  %tmp63 = add nuw nsw i32 %tmp4, 8
  %tmp64 = add i32 %tmp5, 8
  %tmp65 = icmp eq i32 %tmp64, %tmp
  br i1 %tmp65, label %bb1, label %bb2
}
