; RUN: opt %loadPolly -S -polly-codegen -verify-loop-info < %s | FileCheck %s
;
; Check that we do not crash as described here: http://llvm.org/bugs/show_bug.cgi?id=21167
;
; CHECK: polly.split_new_and_old
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind uwtable
define void @kernel_fdtd_apml(i32 %cxm, i32 %cym, ptr %Bza, ptr %Hz, ptr %czp) #0 {
entry:
  br i1 false, label %for.cond4.preheader, label %for.end451

for.cond4.preheader:                              ; preds = %for.inc449, %entry
  %iz.08 = phi i32 [ undef, %for.inc449 ], [ 0, %entry ]
  %cmp55 = icmp sgt i32 %cym, 0
  br i1 %cmp55, label %for.cond7.preheader, label %for.inc449

for.cond7.preheader:                              ; preds = %for.end, %for.cond4.preheader
  %iy.06 = phi i32 [ %inc447, %for.end ], [ 0, %for.cond4.preheader ]
  %cmp81 = icmp sgt i32 %cxm, 0
  br i1 %cmp81, label %for.body9, label %for.end

for.body9:                                        ; preds = %for.body9, %for.cond7.preheader
  %ix.02 = phi i32 [ %inc, %for.body9 ], [ 0, %for.cond7.preheader ]
  %idxprom74 = sext i32 %iz.08 to i64
  %arrayidx75 = getelementptr inbounds double, ptr %czp, i64 %idxprom74
  %0 = load double, ptr %arrayidx75, align 8
  %idxprom102 = sext i32 %iz.08 to i64
  %arrayidx105 = getelementptr inbounds [65 x [65 x double]], ptr %Hz, i64 %idxprom102, i64 0, i64 0
  store double undef, ptr %arrayidx105, align 8
  %inc = add nsw i32 %ix.02, 1
  br i1 false, label %for.body9, label %for.end

for.end:                                          ; preds = %for.body9, %for.cond7.preheader
  %idxprom209 = sext i32 %cxm to i64
  %idxprom211 = sext i32 %iz.08 to i64
  %arrayidx214 = getelementptr inbounds [65 x [65 x double]], ptr %Hz, i64 %idxprom211, i64 0, i64 %idxprom209
  store double undef, ptr %arrayidx214, align 8
  %idxprom430 = sext i32 %cxm to i64
  %idxprom431 = sext i32 %cym to i64
  %idxprom432 = sext i32 %iz.08 to i64
  %arrayidx435 = getelementptr inbounds [65 x [65 x double]], ptr %Hz, i64 %idxprom432, i64 %idxprom431, i64 %idxprom430
  store double undef, ptr %arrayidx435, align 8
  store double undef, ptr %Bza, align 8
  %inc447 = add nsw i32 %iy.06, 1
  %cmp5 = icmp slt i32 %inc447, %cym
  br i1 %cmp5, label %for.cond7.preheader, label %for.inc449

for.inc449:                                       ; preds = %for.end, %for.cond4.preheader
  br i1 undef, label %for.cond4.preheader, label %for.end451

for.end451:                                       ; preds = %for.inc449, %entry
  ret void
}
