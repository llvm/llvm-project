;RUN: opt -opaque-pointers=0 %loadPolly -polly-import-jscop -polly-import-jscop-postfix=transformed -polly-codegen -instnamer < %s -S | FileCheck %s
;
;float A[100];
;
;int codegen_simple () {
;  for (int i = 0; i < 12; i++)
;    A[13] = A[i] + A[i-1];
;
;  return 0;
;}
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32"

@A = common global [100 x float] zeroinitializer, align 4

define i32 @codegen_simple() nounwind {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %tmp1 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %tmp = add i32 %tmp1, -1
  %arrayidx4 = getelementptr [100 x float], [100 x float]* @A, i32 0, i32 %tmp
  %arrayidx = getelementptr [100 x float], [100 x float]* @A, i32 0, i32 %tmp1
  %exitcond = icmp ne i32 %tmp1, 12
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %tmp2 = load float, float* %arrayidx, align 4
  %tmp5 = load float, float* %arrayidx4, align 4
  %add = fadd float %tmp2, %tmp5
  store float %add, float* getelementptr inbounds ([100 x float], [100 x float]* @A, i32 0, i32 13), align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nsw i32 %tmp1, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret i32 0
}
; CHECK: load float, float* getelementptr inbounds ([100 x float], [100 x float]* @A, i{{(32|64)}} 0, i{{(32|64)}} 0)
