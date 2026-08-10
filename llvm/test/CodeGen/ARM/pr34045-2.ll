; RUN: llc < %s -mtriple thumbv7 | FileCheck %s

define hidden void @foo(ptr %ptr, i1 zeroext %long_blocks) {
entry:
; This test is actually checking that no cycle is introduced but at least we
; want to see one umull.
; CHECK: umull
  %0 = load i32, ptr %ptr, align 4
  %conv.i.i13.i = zext i32 %0 to i64
  %mul.i.i14.i = mul nuw nsw i64 %conv.i.i13.i, 18782
  %1 = load i32, ptr undef, align 4
  %conv4.i.i16.i = zext i32 %1 to i64
  %add5.i.i17.i = add nuw nsw i64 %mul.i.i14.i, %conv4.i.i16.i
  %shr.i.i18.i = lshr i64 %add5.i.i17.i, 32
  %add10.i.i20.i = add nuw nsw i64 %shr.i.i18.i, %add5.i.i17.i
  %conv11.i.i21.i = trunc i64 %add10.i.i20.i to i32
  %x.0.neg.i.i26.i = sub i32 -2, %conv11.i.i21.i
  %sub.i.i27.i = add i32 %x.0.neg.i.i26.i, 0
  store i32 %sub.i.i27.i, ptr %ptr, align 4
  br label %while.body.i

while.body.i:                                     ; preds = %while.body.i, %entry
  br label %while.body.i
}

