; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -disable-output < %s 2>&1 | FileCheck %s
;
; Check that the context is built fast and does not explode due to us
; combining a large number of non-convex ranges. Instead, after a certain
; time, we store range information with reduced precision.
;
; CHECK: Context:
; CHECK:      [tmp_0, tmp_1, tmp_2, tmp_3, tmp_4, tmp_5, tmp_6, tmp_7, tmp_8,
; CHECK:       tmp_9, tmp_10, tmp_11, tmp_12, tmp_13, tmp_14, tmp_15] -> {  :
; CHECK:   -2147483648 <= tmp_0 <= 2147483647 and
; CHECK:   -2147483648 <= tmp_1 <= 2147483647 and
; CHECK:   -2147483648 <= tmp_2 <= 2147483647 and
; CHECK:   -2147483648 <= tmp_3 <= 2147483647 and
; CHECK:   -2147483648 <= tmp_4 <= 2147483647 and
; CHECK:   -2147483648 <= tmp_5 <= 2147483647 and
; CHECK:   -2147483648 <= tmp_6 <= 2147483647 and
; CHECK:   -2147483648 <= tmp_7 <= 2147483647 and
; CHECK:   -2147483648 <= tmp_8 <= 2147483647 and
; CHECK:   -2147483648 <= tmp_9 <= 2147483647 and
; CHECK:   -2147483648 <= tmp_10 <= 2147483647 and
; CHECK:   -2147483648 <= tmp_11 <= 2147483647 and
; CHECK:   -2147483648 <= tmp_12 <= 2147483647 and
; CHECK:   -2147483648 <= tmp_13 <= 2147483647 and
; CHECK:   -2147483648 <= tmp_14 <= 2147483647 and
; CHECK:   -2147483648 <= tmp_15 <= 2147483647 and
; CHECK:   ((tmp_0 >= 256 and tmp_1 >= 256 and tmp_2 >= 256) or
; CHECK:    (tmp_0 >= 256 and tmp_1 >= 256 and tmp_2 < 0) or
; CHECK:    (tmp_0 >= 256 and tmp_1 < 0 and tmp_2 >= 256) or
; CHECK:    (tmp_0 >= 256 and tmp_1 < 0 and tmp_2 < 0) or
; CHECK:    (tmp_0 < 0 and tmp_1 >= 256 and tmp_2 >= 256) or
; CHECK:    (tmp_0 < 0 and tmp_1 >= 256 and tmp_2 < 0) or
; CHECK:    (tmp_0 < 0 and tmp_1 < 0 and tmp_2 >= 256) or
; CHECK:    (tmp_0 < 0 and tmp_1 < 0 and tmp_2 < 0)) }
;
;    void jd(int *A, int *p /* in [256, 0) */) {
;      for (int i = 0; i < 1024; i++)
;        A[i + *p] = i;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @jd(ptr %A,
  ptr %p_0,
  ptr %p_1,
  ptr %p_2,
  ptr %p_3,
  ptr %p_4,
  ptr %p_5,
  ptr %p_6,
  ptr %p_7,
  ptr %p_8,
  ptr %p_9,
  ptr %p_10,
  ptr %p_11,
  ptr %p_12,
  ptr %p_13,
  ptr %p_14,
  ptr %p_15
  ) {
entry:
  %tmp_0 = load i32, ptr %p_0, !range !0
  %tmp_1 = load i32, ptr %p_1, !range !0
  %tmp_2 = load i32, ptr %p_2, !range !0
  %tmp_3 = load i32, ptr %p_3, !range !0
  %tmp_4 = load i32, ptr %p_4, !range !0
  %tmp_5 = load i32, ptr %p_5, !range !0
  %tmp_6 = load i32, ptr %p_6, !range !0
  %tmp_7 = load i32, ptr %p_7, !range !0
  %tmp_8 = load i32, ptr %p_8, !range !0
  %tmp_9 = load i32, ptr %p_9, !range !0
  %tmp_10 = load i32, ptr %p_10, !range !0
  %tmp_11 = load i32, ptr %p_11, !range !0
  %tmp_12 = load i32, ptr %p_12, !range !0
  %tmp_13 = load i32, ptr %p_13, !range !0
  %tmp_14 = load i32, ptr %p_14, !range !0
  %tmp_15 = load i32, ptr %p_15, !range !0
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %exitcond = icmp ne i32 %i.0, 1024
  br i1 %exitcond, label %for.body_0, label %for.end

for.body_0:
  %add_0 = add i32 %i.0, %tmp_0
  %idxprom_0 = sext i32 %add_0 to i64
  %arrayidx_0 = getelementptr inbounds i32, ptr %A, i64 %idxprom_0
  store i32 %i.0, ptr %arrayidx_0, align 4
  br label %for.body_1

for.body_1:
  %add_1 = add i32 %i.0, %tmp_1
  %idxprom_1 = sext i32 %add_1 to i64
  %arrayidx_1 = getelementptr inbounds i32, ptr %A, i64 %idxprom_1
  store i32 %i.0, ptr %arrayidx_1, align 4
  br label %for.body_2

for.body_2:
  %add_2 = add i32 %i.0, %tmp_2
  %idxprom_2 = sext i32 %add_2 to i64
  %arrayidx_2 = getelementptr inbounds i32, ptr %A, i64 %idxprom_2
  store i32 %i.0, ptr %arrayidx_2, align 4
  br label %for.body_3

for.body_3:
  %add_3 = add i32 %i.0, %tmp_3
  %idxprom_3 = sext i32 %add_3 to i64
  %arrayidx_3 = getelementptr inbounds i32, ptr %A, i64 %idxprom_3
  store i32 %i.0, ptr %arrayidx_3, align 4
  br label %for.body_4

for.body_4:
  %add_4 = add i32 %i.0, %tmp_4
  %idxprom_4 = sext i32 %add_4 to i64
  %arrayidx_4 = getelementptr inbounds i32, ptr %A, i64 %idxprom_4
  store i32 %i.0, ptr %arrayidx_4, align 4
  br label %for.body_5

for.body_5:
  %add_5 = add i32 %i.0, %tmp_5
  %idxprom_5 = sext i32 %add_5 to i64
  %arrayidx_5 = getelementptr inbounds i32, ptr %A, i64 %idxprom_5
  store i32 %i.0, ptr %arrayidx_5, align 4
  br label %for.body_6

for.body_6:
  %add_6 = add i32 %i.0, %tmp_6
  %idxprom_6 = sext i32 %add_6 to i64
  %arrayidx_6 = getelementptr inbounds i32, ptr %A, i64 %idxprom_6
  store i32 %i.0, ptr %arrayidx_6, align 4
  br label %for.body_7

for.body_7:
  %add_7 = add i32 %i.0, %tmp_7
  %idxprom_7 = sext i32 %add_7 to i64
  %arrayidx_7 = getelementptr inbounds i32, ptr %A, i64 %idxprom_7
  store i32 %i.0, ptr %arrayidx_7, align 4
  br label %for.body_8

for.body_8:
  %add_8 = add i32 %i.0, %tmp_8
  %idxprom_8 = sext i32 %add_8 to i64
  %arrayidx_8 = getelementptr inbounds i32, ptr %A, i64 %idxprom_8
  store i32 %i.0, ptr %arrayidx_8, align 4
  br label %for.body_9

for.body_9:
  %add_9 = add i32 %i.0, %tmp_9
  %idxprom_9 = sext i32 %add_9 to i64
  %arrayidx_9 = getelementptr inbounds i32, ptr %A, i64 %idxprom_9
  store i32 %i.0, ptr %arrayidx_9, align 4
  br label %for.body_10

for.body_10:
  %add_10 = add i32 %i.0, %tmp_10
  %idxprom_10 = sext i32 %add_10 to i64
  %arrayidx_10 = getelementptr inbounds i32, ptr %A, i64 %idxprom_10
  store i32 %i.0, ptr %arrayidx_10, align 4
  br label %for.body_11

for.body_11:
  %add_11 = add i32 %i.0, %tmp_11
  %idxprom_11 = sext i32 %add_11 to i64
  %arrayidx_11 = getelementptr inbounds i32, ptr %A, i64 %idxprom_11
  store i32 %i.0, ptr %arrayidx_11, align 4
  br label %for.body_12

for.body_12:
  %add_12 = add i32 %i.0, %tmp_12
  %idxprom_12 = sext i32 %add_12 to i64
  %arrayidx_12 = getelementptr inbounds i32, ptr %A, i64 %idxprom_12
  store i32 %i.0, ptr %arrayidx_12, align 4
  br label %for.body_13

for.body_13:
  %add_13 = add i32 %i.0, %tmp_13
  %idxprom_13 = sext i32 %add_13 to i64
  %arrayidx_13 = getelementptr inbounds i32, ptr %A, i64 %idxprom_13
  store i32 %i.0, ptr %arrayidx_13, align 4
  br label %for.body_14

for.body_14:
  %add_14 = add i32 %i.0, %tmp_14
  %idxprom_14 = sext i32 %add_14 to i64
  %arrayidx_14 = getelementptr inbounds i32, ptr %A, i64 %idxprom_14
  store i32 %i.0, ptr %arrayidx_14, align 4
  br label %for.body_15

for.body_15:
  %add_15 = add i32 %i.0, %tmp_15
  %idxprom_15 = sext i32 %add_15 to i64
  %arrayidx_15 = getelementptr inbounds i32, ptr %A, i64 %idxprom_15
  store i32 %i.0, ptr %arrayidx_15, align 4
  br label %for.body_end

for.body_end:
  br label %for.inc

for.inc:
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

!0 =  !{ i32 256, i32 0 }
