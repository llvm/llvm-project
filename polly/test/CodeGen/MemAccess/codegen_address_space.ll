;RUN: opt %loadNPMPolly '-passes=polly-import-jscop,polly-codegen' -polly-import-jscop-postfix=transformed < %s -S | FileCheck %s

;int A[100];
;
;int codegen_simple () {
;  for (int i = 0; i < 12; i++)
;    A[13] = A[i] + A[i-1];
;
;  return 0;
;}

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32"

define i32 @codegen_simple(ptr addrspace(5) %A) nounwind {
entry:
  %arrayidx.write = getelementptr [100 x i32], ptr addrspace(5) %A, i32 0, i32 13
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %tmp1 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %tmp = add i32 %tmp1, -1
  %arrayidx4 = getelementptr [100 x i32], ptr addrspace(5) %A, i32 0, i32 %tmp
  %arrayidx = getelementptr [100 x i32], ptr addrspace(5) %A, i32 0, i32 %tmp1
  %exitcond = icmp ne i32 %tmp1, 12
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %tmp2 = load i32, ptr addrspace(5) %arrayidx, align 4
  %tmp5 = load i32, ptr addrspace(5) %arrayidx4, align 4
  %add = add nsw i32 %tmp2, %tmp5
  store i32 %add, ptr addrspace(5) %arrayidx.write
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nsw i32 %tmp1, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret i32 0
}

; CHECK: %polly.access.A = getelementptr i32, ptr addrspace(5) %A, i64 0
; CHECK: %tmp2_p_scalar_ = load i32, ptr addrspace(5) %polly.access.A, align 4, !alias.scope !0, !noalias !3
