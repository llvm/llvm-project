; RUN: opt %loadPolly -polly-print-detect -polly-scops  -disable-output < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; This test case contains two scops.
define void @test(i32 %l, ptr %a) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  %shl = shl i32 %l, 2
  br i1 false, label %for.body, label %for.end

for.body:                                         ; preds = %for.body, %entry.split
  %j.011 = phi i32 [ 0, %entry.split ], [ %add76, %for.body ]
  %add76 = add nsw i32 %j.011, 2
  br i1 false, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry.split
  br i1 undef, label %for.body81, label %for.end170

for.body81:                                       ; preds = %for.body81, %for.end
  %j.19 = phi i32 [ %shl, %for.end ], [ %add169, %for.body81 ]
  %add13710 = or i32 %j.19, 1
  %idxprom138 = sext i32 %add13710 to i64
  %arrayidx139 = getelementptr inbounds double, ptr %a, i64 %idxprom138
  store double undef, ptr %arrayidx139, align 8
  %add169 = add nsw i32 %j.19, 2
  br i1 false, label %for.body81, label %for.end170

for.end170:                                       ; preds = %for.body81
  ret void
}

; CHECK: Valid Region for Scop: entry.split => for.end
; CHECK: Valid Region for Scop: for.body81 => for.end170

