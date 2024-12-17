; RUN: opt -passes=loop-vectorize -force-tail-folding-style=data-with-evl -prefer-predicate-over-epilogue=predicate-dont-vectorize -mtriple=riscv64 -mattr=+v -S %s

; Make sure we don't crash when transforming a VPWidenCastRecipe is created
; without an underlying value. This occurs in this test via
; VPlanTransforms::truncateToMinimalBitwidths

define void @truncate_to_minimal_bitwidths_widen_cast_recipe(ptr %dst, ptr %src, i32 %mvx) {
entry:
  %cmp111 = icmp sgt i32 %mvx, 0
  br i1 %cmp111, label %for.body13.preheader, label %for.cond.cleanup12

for.body13.preheader:                             ; preds = %entry
  %wide.trip.count = zext nneg i32 %mvx to i64
  br label %for.body13

for.body13:                                       ; preds = %for.body13.preheader, %for.body13
  %indvars.iv = phi i64 [ 0, %for.body13.preheader ], [ %indvars.iv.next, %for.body13 ]
  %arrayidx15 = getelementptr i8, ptr %src, i64 %indvars.iv
  %0 = load i8, ptr %arrayidx15, align 1
  %conv = zext i8 %0 to i32
  %mul16 = mul i32 %mvx, %conv
  %shr35 = lshr i32 %mul16, 1
  %conv36 = trunc i32 %shr35 to i8
  store i8 %conv36, ptr %dst, align 1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup12, label %for.body13

for.cond.cleanup12:                               ; preds = %for.body13, %entry
  ret void
}
