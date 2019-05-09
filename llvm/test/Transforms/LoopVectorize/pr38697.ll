; RUN: opt -loop-vectorize -force-vector-width=2 -S < %s 2>&1 | FileCheck %s
; RUN: opt -indvars -S < %s 2>&1 | FileCheck %s -check-prefix=INDVARCHECK

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Produced from test-case:
;
; void testCountIncrLoop(unsigned char *ptr, int lim, int count, int val)
; {
;   int inx = 0;
;   for (int outer_i = 0; outer_i < lim; ++outer_i) {
;     if (count > 0) { // At runtime, 'count' is 0, so the following code is dead.
;       int result = val;
;       int tmp = count;
;
;       while (tmp < 8) {
;         result += val >> tmp;
;         tmp += count;
;       }
;
;       ptr[inx++] = (unsigned char) result;
;     }
;   }
; }
;
; No explicit division appears in the input, but a division is generated during
; vectorization, and that division is a division-by-0 when the input 'count'
; is 0, so it cannot be hoisted above the guard of 'count > 0'.

; Verify that a 'udiv' does not appear in the 'loop1.preheader' block, and that
; a 'udiv' has been inserted at the top of the 'while.body.preheader' block.
define void @testCountIncrLoop(i8* %ptr, i32 %lim, i32 %count, i32 %val) {
; CHECK-LABEL: @testCountIncrLoop(
; CHECK-NEXT:  entry:
; CHECK:       loop1.preheader:
; CHECK-NOT:     udiv
; CHECK:       loop1.body:
; CHECK:       while.cond.preheader:
; CHECK:       while.body.preheader:
; CHECK-NEXT:    [[TMP1:%.*]] = udiv i32 [[TMP0:%.*]], [[COUNT:%.*]]
; CHECK:       vector.ph:
; CHECK:       exit:
; CHECK:         ret void
;
entry:
  %cmp1 = icmp sgt i32 %lim, 0
  br i1 %cmp1, label %loop1.preheader, label %exit

loop1.preheader:                                  ; preds = %entry
  %cmp2 = icmp sgt i32 %count, 0
  %cmp4 = icmp slt i32 %count, 8
  br label %loop1.body

loop1.body:                                       ; preds = %loop1.inc, %loop1.preheader
  %outer_i = phi i32 [ 0, %loop1.preheader ], [ %outer_i.1, %loop1.inc ]
  %inx.1 = phi i32 [ 0, %loop1.preheader ], [ %inx.2, %loop1.inc ]
  br i1 %cmp2, label %while.cond.preheader, label %loop1.inc

while.cond.preheader:                             ; preds = %loop1.body
  br i1 %cmp4, label %while.body, label %while.end

while.body:                                       ; preds = %while.body, %while.cond.preheader
  %tmp = phi i32 [ %add3, %while.body ], [ %count, %while.cond.preheader ]
  %result.1 = phi i32 [ %add, %while.body ], [ %val, %while.cond.preheader ]
  %shr = ashr i32 %val, %tmp
  %add = add nsw i32 %shr, %result.1
  %add3 = add nsw i32 %tmp, %count
  %cmp3 = icmp slt i32 %add3, 8
  br i1 %cmp3, label %while.body, label %while.end

while.end:                                        ; preds = %while.body, %while.cond.preheader
  %result.0.lcssa = phi i32 [ %val, %while.cond.preheader ], [ %add, %while.body ]
  %conv = trunc i32 %result.0.lcssa to i8
  %inc = add nsw i32 %inx.1, 1
  %idxprom = sext i32 %inx.1 to i64
  %arrayidx = getelementptr inbounds i8, i8* %ptr, i64 %idxprom
  store i8 %conv, i8* %arrayidx, align 1
  br label %loop1.inc

loop1.inc:                                        ; preds = %while.end, %loop1.body
  %inx.2 = phi i32 [ %inc, %while.end ], [ %inx.1, %loop1.body ]
  %outer_i.1 = add nuw nsw i32 %outer_i, 1
  %exitcond = icmp eq i32 %outer_i.1, %lim
  br i1 %exitcond, label %exit, label %loop1.body

exit:                                             ; preds = %loop1.inc, %entry
  ret void
}

; These next tests are all based on the following source code, with slight
; variations on the calculation of 'incr' (all of which are loop-invariant
; divisions, but only some of which can be safely hoisted):
;
; uint32_t foo(uint32_t *ptr, uint32_t start1, uint32_t start2) {
;   uint32_t counter1, counter2;
;   uint32_t val = start1;
;   for (counter1 = 1; counter1 < 100; ++counter1) {
;     uint32_t index = 0;
;     val += ptr[index];
;     for (counter2 = start2; counter2 < 10; ++counter2) {
;       // Division is loop invariant, and denominator is guaranteed non-zero:
;       // Safe to hoist it out of the inner loop.
;       uint32_t incr = 16 / counter1;
;       index += incr;
;       val += ptr[index];
;     }
;   }
;   return val;
; }

; This version is as written above, where 'incr' is '16/counter1', and it is
; guaranted that 'counter1' is always non-zero.  So it is safe to hoist the
; division from the inner loop to the preheader.
;
; Verify that the 'udiv' is hoisted to the preheader, and is not in the loop body.
define i32 @NonZeroDivHoist(i32* nocapture readonly %ptr, i32 %start1, i32 %start2) {
; INDVARCHECK-LABEL: @NonZeroDivHoist(
; INDVARCHECK-NEXT:  entry:
; INDVARCHECK:       for.body3.lr.ph:
; INDVARCHECK-NEXT:    [[TMP0:%.*]] = udiv i64 16, [[INDVARS_IV:%.*]]
; INDVARCHECK-NEXT:    br label [[FOR_BODY3:%.*]]
; INDVARCHECK:       for.body3:
; INDVARCHECK-NOT:     udiv
; INDVARCHECK:       for.end10:
;
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.end, %entry
  %val.0 = phi i32 [ %start1, %entry ], [ %val.1.lcssa, %for.end ]
  %counter1.0 = phi i32 [ 1, %entry ], [ %inc9, %for.end ]
  %cmp = icmp ult i32 %counter1.0, 100
  br i1 %cmp, label %for.body, label %for.end10

for.body:                                         ; preds = %for.cond
  %tmp = load i32, i32* %ptr, align 4
  %add = add i32 %tmp, %val.0
  %cmp224 = icmp ult i32 %start2, 10
  br i1 %cmp224, label %for.body3.lr.ph, label %for.end

for.body3.lr.ph:                                  ; preds = %for.body
  br label %for.body3

for.body3:                                        ; preds = %for.body3, %for.body3.lr.ph
  %index.027 = phi i32 [ 0, %for.body3.lr.ph ], [ %add4, %for.body3 ]
  %val.126 = phi i32 [ %add, %for.body3.lr.ph ], [ %add7, %for.body3 ]
  %counter2.025 = phi i32 [ %start2, %for.body3.lr.ph ], [ %inc, %for.body3 ]
  %div = udiv i32 16, %counter1.0
  %add4 = add i32 %div, %index.027
  %idxprom5 = zext i32 %add4 to i64
  %arrayidx6 = getelementptr inbounds i32, i32* %ptr, i64 %idxprom5
  %tmp1 = load i32, i32* %arrayidx6, align 4
  %add7 = add i32 %tmp1, %val.126
  %inc = add i32 %counter2.025, 1
  %cmp2 = icmp ult i32 %inc, 10
  br i1 %cmp2, label %for.body3, label %for.cond1.for.end_crit_edge

for.cond1.for.end_crit_edge:                      ; preds = %for.body3
  %split = phi i32 [ %add7, %for.body3 ]
  br label %for.end

for.end:                                          ; preds = %for.cond1.for.end_crit_edge, %for.body
  %val.1.lcssa = phi i32 [ %split, %for.cond1.for.end_crit_edge ], [ %add, %for.body ]
  %inc9 = add i32 %counter1.0, 1
  br label %for.cond

for.end10:                                        ; preds = %for.cond
  %val.0.lcssa = phi i32 [ %val.0, %for.cond ]
  ret i32 %val.0.lcssa
}

; This version is identical to the above 'NonZeroDivHoist' case, except the
; outer ('counter1') loop starts at the unknown value of 'start1' rather than 1,
; and so it is illegal to hoist the division because if 'start1' is 0, hoisting
; it would incorrectly cause a divide-by-zero trap.
;
; Verify that the 'udiv' is not hoisted to the preheader, and it remains in the
; loop body.
define i32 @ZeroDivNoHoist(i32* nocapture readonly %ptr, i32 %start1, i32 %start2) {
; INDVARCHECK-LABEL: @ZeroDivNoHoist(
; INDVARCHECK-NEXT:  entry:
; INDVARCHECK-NOT:     udiv
; INDVARCHECK:       for.body3:
; INDVARCHECK:         [[TMP1:%.*]] = udiv i64 16, [[INDVARS_IV:%.*]]
; INDVARCHECK:       for.cond1.for.end_crit_edge:
;
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.end, %entry
  %val.0 = phi i32 [ %start1, %entry ], [ %val.1.lcssa, %for.end ]
  %counter1.0 = phi i32 [ %start1, %entry ], [ %inc9, %for.end ]
  %cmp = icmp ult i32 %counter1.0, 100
  br i1 %cmp, label %for.body, label %for.end10

for.body:                                         ; preds = %for.cond
  %tmp = load i32, i32* %ptr, align 4
  %add = add i32 %tmp, %val.0
  %cmp224 = icmp ult i32 %start2, 10
  br i1 %cmp224, label %for.body3.lr.ph, label %for.end

for.body3.lr.ph:                                  ; preds = %for.body
  br label %for.body3

for.body3:                                        ; preds = %for.body3, %for.body3.lr.ph
  %index.027 = phi i32 [ 0, %for.body3.lr.ph ], [ %add4, %for.body3 ]
  %val.126 = phi i32 [ %add, %for.body3.lr.ph ], [ %add7, %for.body3 ]
  %counter2.025 = phi i32 [ %start2, %for.body3.lr.ph ], [ %inc, %for.body3 ]
  %div = udiv i32 16, %counter1.0
  %add4 = add i32 %div, %index.027
  %idxprom5 = zext i32 %add4 to i64
  %arrayidx6 = getelementptr inbounds i32, i32* %ptr, i64 %idxprom5
  %tmp1 = load i32, i32* %arrayidx6, align 4
  %add7 = add i32 %tmp1, %val.126
  %inc = add i32 %counter2.025, 1
  %cmp2 = icmp ult i32 %inc, 10
  br i1 %cmp2, label %for.body3, label %for.cond1.for.end_crit_edge

for.cond1.for.end_crit_edge:                      ; preds = %for.body3
  %split = phi i32 [ %add7, %for.body3 ]
  br label %for.end

for.end:                                          ; preds = %for.cond1.for.end_crit_edge, %for.body
  %val.1.lcssa = phi i32 [ %split, %for.cond1.for.end_crit_edge ], [ %add, %for.body ]
  %inc9 = add i32 %counter1.0, 1
  br label %for.cond

for.end10:                                        ; preds = %for.cond
  %val.0.lcssa = phi i32 [ %val.0, %for.cond ]
  ret i32 %val.0.lcssa
}

; This version is has a clearly safe division by a non-zero constant (16).  The
; division is transformed to a logical-shift-right of 4, and it is safely
; hoisted.
;
; Verify that the division-operation is hoisted, and that it appears as a
; right-shift ('lshr') rather than an explicit division.
define i32 @DivBy16Hoist(i32* nocapture readonly %ptr, i32 %start1, i32 %start2) {
; INDVARCHECK-LABEL: @DivBy16Hoist(
; INDVARCHECK-NEXT:  entry:
; INDVARCHECK:       for.cond:
; INDVARCHECK:         [[TMP1:%.*]] = lshr i64 [[INDVARS_IV:%.*]], 4
; INDVARCHECK:       for.body:
; INDVARCHECK-NOT:     lshr
; INDVARCHECK-NOT:     udiv
; INDVARCHECK:       for.end10:
;
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.end, %entry
  %val.0 = phi i32 [ %start1, %entry ], [ %val.1.lcssa, %for.end ]
  %counter1.0 = phi i32 [ %start1, %entry ], [ %inc9, %for.end ]
  %cmp = icmp ult i32 %counter1.0, 100
  br i1 %cmp, label %for.body, label %for.end10

for.body:                                         ; preds = %for.cond
  %tmp = load i32, i32* %ptr, align 4
  %add = add i32 %tmp, %val.0
  %cmp224 = icmp ult i32 %start2, 10
  br i1 %cmp224, label %for.body3.lr.ph, label %for.end

for.body3.lr.ph:                                  ; preds = %for.body
  br label %for.body3

for.body3:                                        ; preds = %for.body3, %for.body3.lr.ph
  %index.027 = phi i32 [ 0, %for.body3.lr.ph ], [ %add4, %for.body3 ]
  %val.126 = phi i32 [ %add, %for.body3.lr.ph ], [ %add7, %for.body3 ]
  %counter2.025 = phi i32 [ %start2, %for.body3.lr.ph ], [ %inc, %for.body3 ]
  %div = udiv i32 %counter1.0, 16
  %add4 = add i32 %div, %index.027
  %idxprom5 = zext i32 %add4 to i64
  %arrayidx6 = getelementptr inbounds i32, i32* %ptr, i64 %idxprom5
  %tmp1 = load i32, i32* %arrayidx6, align 4
  %add7 = add i32 %tmp1, %val.126
  %inc = add i32 %counter2.025, 1
  %cmp2 = icmp ult i32 %inc, 10
  br i1 %cmp2, label %for.body3, label %for.cond1.for.end_crit_edge

for.cond1.for.end_crit_edge:                      ; preds = %for.body3
  %split = phi i32 [ %add7, %for.body3 ]
  br label %for.end

for.end:                                          ; preds = %for.cond1.for.end_crit_edge, %for.body
  %val.1.lcssa = phi i32 [ %split, %for.cond1.for.end_crit_edge ], [ %add, %for.body ]
  %inc9 = add i32 %counter1.0, 1
  br label %for.cond

for.end10:                                        ; preds = %for.cond
  %val.0.lcssa = phi i32 [ %val.0, %for.cond ]
  ret i32 %val.0.lcssa
}

; This version is has a clearly safe division by a non-zero constant (17).  The
; division is safely hoisted, as it was in the 'DivBy16Hoist' verison, but here
; it remains a division, rather than being transformed to a right-shift.
;
; Verify that the division-operation is hoisted.
define i32 @DivBy17Hoist(i32* nocapture readonly %ptr, i32 %start1, i32 %start2) {
; INDVARCHECK-LABEL: @DivBy17Hoist(
; INDVARCHECK-NEXT:  entry:
; INDVARCHECK:       for.cond:
; INDVARCHECK:         [[TMP1:%.*]] = udiv i64 [[INDVARS_IV:%.*]], 17
; INDVARCHECK:       for.body:
; INDVARCHECK-NOT:     udiv
; INDVARCHECK:       for.end10:
;
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.end, %entry
  %val.0 = phi i32 [ %start1, %entry ], [ %val.1.lcssa, %for.end ]
  %counter1.0 = phi i32 [ %start1, %entry ], [ %inc9, %for.end ]
  %cmp = icmp ult i32 %counter1.0, 100
  br i1 %cmp, label %for.body, label %for.end10

for.body:                                         ; preds = %for.cond
  %tmp = load i32, i32* %ptr, align 4
  %add = add i32 %tmp, %val.0
  %cmp224 = icmp ult i32 %start2, 10
  br i1 %cmp224, label %for.body3.lr.ph, label %for.end

for.body3.lr.ph:                                  ; preds = %for.body
  br label %for.body3

for.body3:                                        ; preds = %for.body3, %for.body3.lr.ph
  %index.027 = phi i32 [ 0, %for.body3.lr.ph ], [ %add4, %for.body3 ]
  %val.126 = phi i32 [ %add, %for.body3.lr.ph ], [ %add7, %for.body3 ]
  %counter2.025 = phi i32 [ %start2, %for.body3.lr.ph ], [ %inc, %for.body3 ]
  %div = udiv i32 %counter1.0, 17
  %add4 = add i32 %div, %index.027
  %idxprom5 = zext i32 %add4 to i64
  %arrayidx6 = getelementptr inbounds i32, i32* %ptr, i64 %idxprom5
  %tmp1 = load i32, i32* %arrayidx6, align 4
  %add7 = add i32 %tmp1, %val.126
  %inc = add i32 %counter2.025, 1
  %cmp2 = icmp ult i32 %inc, 10
  br i1 %cmp2, label %for.body3, label %for.cond1.for.end_crit_edge

for.cond1.for.end_crit_edge:                      ; preds = %for.body3
  %split = phi i32 [ %add7, %for.body3 ]
  br label %for.end

for.end:                                          ; preds = %for.cond1.for.end_crit_edge, %for.body
  %val.1.lcssa = phi i32 [ %split, %for.cond1.for.end_crit_edge ], [ %add, %for.body ]
  %inc9 = add i32 %counter1.0, 1
  br label %for.cond

for.end10:                                        ; preds = %for.cond
  %val.0.lcssa = phi i32 [ %val.0, %for.cond ]
  ret i32 %val.0.lcssa
}
