; RUN: opt %loadPolly -polly-tile-sizes=1,64 -polly-opt-isl -polly-print-ast -disable-output < %s | FileCheck %s

; CHECK: for (int c0 = 0; c0 <= 1023; c0 += 1)
; CHECK:   for (int c1 = 0; c1 <= 7; c1 += 1)
; CHECK:     for (int c3 = 0; c3 <= 63; c3 += 1)
; CHECK:       Stmt_for_body3(c0, 64 * c1 + c3);

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

; Function Attrs: nounwind
define void @line([512 x i32]* %A) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.body3.lr.ph

for.body3.lr.ph:                                  ; preds = %for.inc5, %entry.split
  %i.0 = phi i32 [ 0, %entry.split ], [ %inc6, %for.inc5 ]
  br label %for.body3

for.body3:                                        ; preds = %for.body3.lr.ph, %for.body3
  %j.0 = phi i32 [ 0, %for.body3.lr.ph ], [ %inc, %for.body3 ]
  %mul = mul nsw i32 %j.0, %i.0
  %rem = srem i32 %mul, 42
  %arrayidx4 = getelementptr inbounds [512 x i32], [512 x i32]* %A, i32 %i.0, i32 %j.0
  store i32 %rem, i32* %arrayidx4, align 4
  %inc = add nsw i32 %j.0, 1
  %cmp2 = icmp slt i32 %inc, 512
  br i1 %cmp2, label %for.body3, label %for.inc5

for.inc5:                                         ; preds = %for.body3
  %inc6 = add nsw i32 %i.0, 1
  %cmp = icmp slt i32 %inc6, 1024
  br i1 %cmp, label %for.body3.lr.ph, label %for.end7

for.end7:                                         ; preds = %for.inc5
  ret void
}
