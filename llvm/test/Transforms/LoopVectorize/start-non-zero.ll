; RUN: opt < %s -passes=loop-vectorize,instcombine -force-vector-interleave=1 -force-vector-width=4 -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

;CHECK-LABEL: @start_at_nonzero(
;CHECK: mul nuw <4 x i32>
;CHECK: ret i32
define i32 @start_at_nonzero(ptr nocapture %a, i32 %start, i32 %end) nounwind uwtable ssp {
entry:
  %cmp3 = icmp slt i32 %start, %end
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  %0 = sext i32 %start to i64
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %indvars.iv = phi i64 [ %0, %for.body.lr.ph ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %indvars.iv
  %1 = load i32, ptr %arrayidx, align 4
  %mul = mul nuw i32 %1, 333
  store i32 %mul, ptr %arrayidx, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %2 = trunc i64 %indvars.iv.next to i32
  %cmp = icmp slt i32 %2, %end
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret i32 4
}
