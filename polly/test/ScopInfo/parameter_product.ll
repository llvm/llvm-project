; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -disable-output < %s 2>&1 | FileCheck %s
;
; int n, m;
; void foo(char* __restrict a)
; {
;   for (int i = 0; i < n*m; ++i)
;     a[i]=2;
; }
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

define void @foo(i8* noalias %a, i32 %param_n, i32 %param_m) {
entry:
  %mul = mul nsw i32 %param_m, %param_n
  %cmp1 = icmp sgt i32 %mul, 0
  br i1 %cmp1, label %for.body, label %for.end

for.body:
  %i.02 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i8, i8* %a, i64 0
  store i8 2, i8* %arrayidx, align 1
  %add = add nsw i32 %i.02, 1
  %cmp = icmp slt i32 %add, %mul
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}

; CHECK: p0: (%param_n * %param_m)

