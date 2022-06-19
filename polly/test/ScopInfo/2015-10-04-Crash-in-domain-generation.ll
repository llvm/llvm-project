; RUN: opt %loadPolly -polly-allow-nonaffine-loops -polly-scops -disable-output < %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind uwtable
define void @kernel_reg_detect([6 x i32]* %path) {
entry:
  br label %for.body.6

for.body.6:                                       ; preds = %for.inc.43, %for.body.6, %entry
  %indvars.iv9 = phi i64 [ %indvars.iv.next10, %for.body.6 ], [ 0, %entry ]
  %indvars.iv.next10 = add nuw nsw i64 %indvars.iv9, 1
  %exitcond = icmp ne i64 %indvars.iv.next10, 6
  br i1 %exitcond, label %for.body.6, label %for.inc.40

for.inc.40:                                       ; preds = %for.inc.40, %for.body.6
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc.40 ], [ 0, %for.body.6 ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %arrayidx28 = getelementptr inbounds [6 x i32], [6 x i32]* %path, i64 0, i64 0
  %tmp = load i32, i32* %arrayidx28, align 4
  %arrayidx36 = getelementptr inbounds [6 x i32], [6 x i32]* %path, i64 0, i64 0
  store i32 0, i32* %arrayidx36, align 4
  %mul = mul i64 %indvars.iv, %indvars.iv
  %exitcond22 = icmp ne i64 %mul, 6
  br i1 %exitcond22, label %for.inc.40, label %for.inc.43

for.inc.43:                                       ; preds = %for.inc.40
  br label %for.end.45

for.end.45:                                       ; preds = %for.inc.43
  ret void
}
