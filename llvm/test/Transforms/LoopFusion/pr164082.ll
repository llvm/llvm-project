; REQUIRES: asserts
; RUN: opt -passes=loop-fusion -disable-output -stats < %s 2>&1 | FileCheck -check-prefix=STAT %s
; STAT: 1 loop-fusion - Loops fused

; XFAIL: *
; Currently fails since delinearization doesn't work as expected. The estimated
; array size is different for `Array[i][i]` and `Array[i][j]`. The former is
; now regarded as an access to a 1D array.

; C Code
;
;;  for (int i = 0; i < 100; ++i)
;;      Array[i][i] = -i;
;;  for (int row = 0; row < 100; ++row)
;;      for (int col = 0; col < 100; ++col)
;;          if (col != row)
;;              Array[row][col] = row + col;
;
; Loop fusion should not crash anymore as now forgetBlockAndLoopDispositions()
; is trigerred after mergeLatch() during the fusion.

define i32 @forget_dispositions() nounwind {
entry:
  %Array = alloca [100 x [100 x i32]], align 4
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv33 = phi i64 [ 0, %entry ], [ %indvars.iv.next34, %for.body ]
  %0 = trunc i64 %indvars.iv33 to i32
  %sub = sub i32 0, %0
  %arrayidx2 = getelementptr inbounds [100 x [100 x i32]], ptr %Array, i64 0, i64 %indvars.iv33, i64 %indvars.iv33
  store i32 %sub, ptr %arrayidx2, align 4
  %indvars.iv.next34 = add i64 %indvars.iv33, 1
  %lftr.wideiv35 = trunc i64 %indvars.iv.next34 to i32
  %exitcond36 = icmp eq i32 %lftr.wideiv35, 100
  br i1 %exitcond36, label %for.cond6.preheader, label %for.body

for.cond6.preheader:                              ; preds = %for.body, %for.inc17
  %indvars.iv29 = phi i64 [ %indvars.iv.next30, %for.inc17 ], [ 0, %for.body ]
  br label %for.body8

for.body8:                                        ; preds = %for.inc14, %for.cond6.preheader
  %indvars.iv = phi i64 [ 0, %for.cond6.preheader ], [ %indvars.iv.next, %for.inc14 ]
  %1 = trunc i64 %indvars.iv to i32
  %2 = trunc i64 %indvars.iv29 to i32
  %cmp9 = icmp eq i32 %1, %2
  br i1 %cmp9, label %for.inc14, label %if.then

if.then:                                          ; preds = %for.body8
  %3 = add i64 %indvars.iv, %indvars.iv29
  %arrayidx13 = getelementptr inbounds [100 x [100 x i32]], ptr %Array, i64 0, i64 %indvars.iv29, i64 %indvars.iv
  %4 = trunc i64 %3 to i32
  store i32 %4, ptr %arrayidx13, align 4
  br label %for.inc14

for.inc14:                                        ; preds = %for.body8, %if.then
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv27 = trunc i64 %indvars.iv.next to i32
  %exitcond28 = icmp eq i32 %lftr.wideiv27, 100
  br i1 %exitcond28, label %for.inc17, label %for.body8

for.inc17:                                        ; preds = %for.inc14
  %indvars.iv.next30 = add i64 %indvars.iv29, 1
  %lftr.wideiv31 = trunc i64 %indvars.iv.next30 to i32
  %exitcond32 = icmp eq i32 %lftr.wideiv31, 100
  br i1 %exitcond32, label %for.exit, label %for.cond6.preheader

for.exit:                                    ; preds = %for.inc17
  ret i32 0
}
