; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -mtriple aarch64-linux-gnu -mattr=+sve2 -sve-tail-folding=simple -debug-only=loop-vectorize -S 2>%t < %s
; RUN: cat %t | FileCheck %s --check-prefix=CHECK-COST

target triple = "aarch64-unknown-linux-gnu"

; SRHADD

define void @srhadd_i8_zext_i16(ptr noalias nocapture %a, ptr noalias nocapture readonly %b, ptr noalias nocapture readonly %dst, i64 %n) {

; CHECK-COST: LV: Found an estimated cost of 0 for VF 1 For instruction:   %sext1 = sext i8 %ld1 to i16
; CHECK-COST: LV: Found an estimated cost of 0 for VF 1 For instruction:   %sext2 = sext i8 %ld2 to i16

; CHECK-COST: LV: Found an estimated cost of 0 for VF 2 For instruction:   %sext1 = sext i8 %ld1 to i16
; CHECK-COST: LV: Found an estimated cost of 0 for VF 2 For instruction:   %sext2 = sext i8 %ld2 to i16

; CHECK-COST: LV: Found an estimated cost of 0 for VF 4 For instruction:   %sext1 = sext i8 %ld1 to i16
; CHECK-COST: LV: Found an estimated cost of 0 for VF 4 For instruction:   %sext2 = sext i8 %ld2 to i16

; CHECK-COST: LV: Found an estimated cost of 0 for VF 8 For instruction:   %sext1 = sext i8 %ld1 to i16
; CHECK-COST: LV: Found an estimated cost of 0 for VF 8 For instruction:   %sext2 = sext i8 %ld2 to i16

; CHECK-COST: LV: Found an estimated cost of 0 for VF 16 For instruction:   %sext1 = sext i8 %ld1 to i16
; CHECK-COST: LV: Found an estimated cost of 0 for VF 16 For instruction:   %sext2 = sext i8 %ld2 to i16

; CHECK-COST: LV: Found an estimated cost of 0 for VF vscale x 1 For instruction:   %sext1 = sext i8 %ld1 to i16
; CHECK-COST: LV: Found an estimated cost of 0 for VF vscale x 1 For instruction:   %sext2 = sext i8 %ld2 to i16

; CHECK-COST: LV: Found an estimated cost of 0 for VF vscale x 2 For instruction:   %sext1 = sext i8 %ld1 to i16
; CHECK-COST: LV: Found an estimated cost of 0 for VF vscale x 2 For instruction:   %sext2 = sext i8 %ld2 to i16

; CHECK-COST: LV: Found an estimated cost of 0 for VF vscale x 4 For instruction:   %sext1 = sext i8 %ld1 to i16
; CHECK-COST: LV: Found an estimated cost of 0 for VF vscale x 4 For instruction:   %sext2 = sext i8 %ld2 to i16

; CHECK-COST: LV: Found an estimated cost of 0 for VF vscale x 8 For instruction:   %sext1 = sext i8 %ld1 to i16
; CHECK-COST: LV: Found an estimated cost of 0 for VF vscale x 8 For instruction:   %sext2 = sext i8 %ld2 to i16

; CHECK-COST: LV: Found an estimated cost of 0 for VF vscale x 16 For instruction:   %sext1 = sext i8 %ld1 to i16
; CHECK-COST: LV: Found an estimated cost of 0 for VF vscale x 16 For instruction:   %sext2 = sext i8 %ld2 to i16

entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx1 = getelementptr inbounds i8, ptr %a, i64 %indvars.iv
  %ld1 = load i8, ptr %arrayidx1
  %sext1 = sext i8 %ld1 to i16
  %arrayidx2 = getelementptr inbounds i8, ptr %b, i64 %indvars.iv
  %ld2 = load i8, ptr %arrayidx2
  %sext2 = sext i8 %ld2 to i16
  %add1 = add nuw nsw i16 %sext1, 1
  %add2 = add nuw nsw i16 %add1, %sext2
  %shr = lshr i16 %add2, 1
  %trunc = trunc i16 %shr to i8
  %arrayidx3 = getelementptr inbounds i8, ptr %dst, i64 %indvars.iv
  store i8 %trunc, ptr %arrayidx3
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %for.end.loopexit, label %for.body

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}

define void @srhadd_i16_zext_i32(ptr noalias nocapture %a, ptr noalias nocapture readonly %b, ptr noalias nocapture readonly %dst, i64 %n) {

; CHECK-COST: LV: Found an estimated cost of 0 for VF 1 For instruction:   %sext1 = sext i16 %ld1 to i32
; CHECK-COST: LV: Found an estimated cost of 0 for VF 1 For instruction:   %sext2 = sext i16 %ld2 to i32

; CHECK-COST: LV: Found an estimated cost of 0 for VF 2 For instruction:   %sext1 = sext i16 %ld1 to i32
; CHECK-COST: LV: Found an estimated cost of 0 for VF 2 For instruction:   %sext2 = sext i16 %ld2 to i32

; CHECK-COST: LV: Found an estimated cost of 0 for VF 4 For instruction:   %sext1 = sext i16 %ld1 to i32
; CHECK-COST: LV: Found an estimated cost of 0 for VF 4 For instruction:   %sext2 = sext i16 %ld2 to i32

; CHECK-COST: LV: Found an estimated cost of 0 for VF 8 For instruction:   %sext1 = sext i16 %ld1 to i32
; CHECK-COST: LV: Found an estimated cost of 0 for VF 8 For instruction:   %sext2 = sext i16 %ld2 to i32

; CHECK-COST: LV: Found an estimated cost of 0 for VF vscale x 1 For instruction:   %sext1 = sext i16 %ld1 to i32
; CHECK-COST: LV: Found an estimated cost of 0 for VF vscale x 1 For instruction:   %sext2 = sext i16 %ld2 to i32

; CHECK-COST: LV: Found an estimated cost of 0 for VF vscale x 2 For instruction:   %sext1 = sext i16 %ld1 to i32
; CHECK-COST: LV: Found an estimated cost of 0 for VF vscale x 2 For instruction:   %sext2 = sext i16 %ld2 to i32

; CHECK-COST: LV: Found an estimated cost of 0 for VF vscale x 4 For instruction:   %sext1 = sext i16 %ld1 to i32
; CHECK-COST: LV: Found an estimated cost of 0 for VF vscale x 4 For instruction:   %sext2 = sext i16 %ld2 to i32

; CHECK-COST: LV: Found an estimated cost of 0 for VF vscale x 8 For instruction:   %sext1 = sext i16 %ld1 to i32
; CHECK-COST: LV: Found an estimated cost of 0 for VF vscale x 8 For instruction:   %sext2 = sext i16 %ld2 to i32

entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx1 = getelementptr inbounds i16, ptr %a, i64 %indvars.iv
  %ld1 = load i16, ptr %arrayidx1
  %sext1 = sext i16 %ld1 to i32
  %arrayidx2 = getelementptr inbounds i16, ptr %b, i64 %indvars.iv
  %ld2 = load i16, ptr %arrayidx2
  %sext2 = sext i16 %ld2 to i32
  %add1 = add nuw nsw i32 %sext1, 1
  %add2 = add nuw nsw i32 %add1, %sext2
  %shr = lshr i32 %add2, 1
  %trunc = trunc i32 %shr to i16
  %arrayidx3 = getelementptr inbounds i16, ptr %dst, i64 %indvars.iv
  store i16 %trunc, ptr %arrayidx3
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %for.end.loopexit, label %for.body

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}

; URHADD

define void @urhadd_i8_zext_i16(ptr noalias nocapture %a, ptr noalias nocapture readonly %b, ptr noalias nocapture readonly %dst, i64 %n) {

; CHECK-COST: LV: Found an estimated cost of 0 for VF 1 For instruction:   %zext1 = zext i8 %ld1 to i16
; CHECK-COST: LV: Found an estimated cost of 0 for VF 1 For instruction:   %zext2 = zext i8 %ld2 to i16

; CHECK-COST: LV: Found an estimated cost of 0 for VF 2 For instruction:   %zext1 = zext i8 %ld1 to i16
; CHECK-COST: LV: Found an estimated cost of 0 for VF 2 For instruction:   %zext2 = zext i8 %ld2 to i16

; CHECK-COST: LV: Found an estimated cost of 0 for VF 4 For instruction:   %zext1 = zext i8 %ld1 to i16
; CHECK-COST: LV: Found an estimated cost of 0 for VF 4 For instruction:   %zext2 = zext i8 %ld2 to i16

; CHECK-COST: LV: Found an estimated cost of 0 for VF 8 For instruction:   %zext1 = zext i8 %ld1 to i16
; CHECK-COST: LV: Found an estimated cost of 0 for VF 8 For instruction:   %zext2 = zext i8 %ld2 to i16

; CHECK-COST: LV: Found an estimated cost of 0 for VF 16 For instruction:   %zext1 = zext i8 %ld1 to i16
; CHECK-COST: LV: Found an estimated cost of 0 for VF 16 For instruction:   %zext2 = zext i8 %ld2 to i16

; CHECK-COST: LV: Found an estimated cost of 0 for VF vscale x 1 For instruction:   %zext1 = zext i8 %ld1 to i16
; CHECK-COST: LV: Found an estimated cost of 0 for VF vscale x 1 For instruction:   %zext2 = zext i8 %ld2 to i16

; CHECK-COST: LV: Found an estimated cost of 0 for VF vscale x 2 For instruction:   %zext1 = zext i8 %ld1 to i16
; CHECK-COST: LV: Found an estimated cost of 0 for VF vscale x 2 For instruction:   %zext2 = zext i8 %ld2 to i16

; CHECK-COST: LV: Found an estimated cost of 0 for VF vscale x 4 For instruction:   %zext1 = zext i8 %ld1 to i16
; CHECK-COST: LV: Found an estimated cost of 0 for VF vscale x 4 For instruction:   %zext2 = zext i8 %ld2 to i16

; CHECK-COST: LV: Found an estimated cost of 0 for VF vscale x 8 For instruction:   %zext1 = zext i8 %ld1 to i16
; CHECK-COST: LV: Found an estimated cost of 0 for VF vscale x 8 For instruction:   %zext2 = zext i8 %ld2 to i16

; CHECK-COST: LV: Found an estimated cost of 0 for VF vscale x 16 For instruction:   %zext1 = zext i8 %ld1 to i16
; CHECK-COST: LV: Found an estimated cost of 0 for VF vscale x 16 For instruction:   %zext2 = zext i8 %ld2 to i16

entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx1 = getelementptr inbounds i8, ptr %a, i64 %indvars.iv
  %ld1 = load i8, ptr %arrayidx1
  %zext1 = zext i8 %ld1 to i16
  %arrayidx2 = getelementptr inbounds i8, ptr %b, i64 %indvars.iv
  %ld2 = load i8, ptr %arrayidx2
  %zext2 = zext i8 %ld2 to i16
  %add1 = add nuw nsw i16 %zext1, 1
  %add2 = add nuw nsw i16 %add1, %zext2
  %shr = lshr i16 %add2, 1
  %trunc = trunc i16 %shr to i8
  %arrayidx3 = getelementptr inbounds i8, ptr %dst, i64 %indvars.iv
  store i8 %trunc, ptr %arrayidx3
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %for.end.loopexit, label %for.body

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}

define void @urhadd_i16_zext_i32(ptr noalias nocapture %a, ptr noalias nocapture readonly %b, ptr noalias nocapture readonly %dst, i64 %n) {

; CHECK-COST: LV: Found an estimated cost of 0 for VF 1 For instruction:   %zext1 = zext i16 %ld1 to i32
; CHECK-COST: LV: Found an estimated cost of 0 for VF 1 For instruction:   %zext2 = zext i16 %ld2 to i32

; CHECK-COST: LV: Found an estimated cost of 0 for VF 2 For instruction:   %zext1 = zext i16 %ld1 to i32
; CHECK-COST: LV: Found an estimated cost of 0 for VF 2 For instruction:   %zext2 = zext i16 %ld2 to i32

; CHECK-COST: LV: Found an estimated cost of 0 for VF 4 For instruction:   %zext1 = zext i16 %ld1 to i32
; CHECK-COST: LV: Found an estimated cost of 0 for VF 4 For instruction:   %zext2 = zext i16 %ld2 to i32

; CHECK-COST: LV: Found an estimated cost of 0 for VF 8 For instruction:   %zext1 = zext i16 %ld1 to i32
; CHECK-COST: LV: Found an estimated cost of 0 for VF 8 For instruction:   %zext2 = zext i16 %ld2 to i32

; CHECK-COST: LV: Found an estimated cost of 0 for VF vscale x 1 For instruction:   %zext1 = zext i16 %ld1 to i32
; CHECK-COST: LV: Found an estimated cost of 0 for VF vscale x 1 For instruction:   %zext2 = zext i16 %ld2 to i32

; CHECK-COST: LV: Found an estimated cost of 0 for VF vscale x 2 For instruction:   %zext1 = zext i16 %ld1 to i32
; CHECK-COST: LV: Found an estimated cost of 0 for VF vscale x 2 For instruction:   %zext2 = zext i16 %ld2 to i32

; CHECK-COST: LV: Found an estimated cost of 0 for VF vscale x 4 For instruction:   %zext1 = zext i16 %ld1 to i32
; CHECK-COST: LV: Found an estimated cost of 0 for VF vscale x 4 For instruction:   %zext2 = zext i16 %ld2 to i32

; CHECK-COST: LV: Found an estimated cost of 0 for VF vscale x 8 For instruction:   %zext1 = zext i16 %ld1 to i32
; CHECK-COST: LV: Found an estimated cost of 0 for VF vscale x 8 For instruction:   %zext2 = zext i16 %ld2 to i32

entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx1 = getelementptr inbounds i16, ptr %a, i64 %indvars.iv
  %ld1 = load i16, ptr %arrayidx1
  %zext1 = zext i16 %ld1 to i32
  %arrayidx2 = getelementptr inbounds i16, ptr %b, i64 %indvars.iv
  %ld2 = load i16, ptr %arrayidx2
  %zext2 = zext i16 %ld2 to i32
  %add1 = add nuw nsw i32 %zext1, 1
  %add2 = add nuw nsw i32 %add1, %zext2
  %shr = lshr i32 %add2, 1
  %trunc = trunc i32 %shr to i16
  %arrayidx3 = getelementptr inbounds i16, ptr %dst, i64 %indvars.iv
  store i16 %trunc, ptr %arrayidx3
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %for.end.loopexit, label %for.body

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}
