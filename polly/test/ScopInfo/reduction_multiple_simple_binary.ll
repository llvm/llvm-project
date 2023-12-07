; RUN: opt -basic-aa %loadPolly -polly-stmt-granularity=bb -polly-print-scops -disable-output < %s | FileCheck %s
;
; CHECK: ReadAccess :=       [Reduction Type: NONE
; CHECK:     { Stmt_for_body[i0] -> MemRef_A[1 + i0] };
; CHECK: ReadAccess :=       [Reduction Type: NONE
; CHECK:     { Stmt_for_body[i0] -> MemRef_A[i0] };
; CHECK: MustWriteAccess :=  [Reduction Type: NONE
; CHECK:     { Stmt_for_body[i0] -> MemRef_first[0] };
; CHECK: ReadAccess :=       [Reduction Type: +
; CHECK:     { Stmt_for_body[i0] -> MemRef_sum[0] };
; CHECK: MustWriteAccess :=  [Reduction Type: +
; CHECK:     { Stmt_for_body[i0] -> MemRef_sum[0] };
; CHECK: ReadAccess :=       [Reduction Type: NONE
; CHECK:     { Stmt_for_body[i0] -> MemRef_A[-1 + i0] };
; CHECK: ReadAccess :=       [Reduction Type: NONE
; CHECK:     { Stmt_for_body[i0] -> MemRef_A[i0] };
; CHECK: MustWriteAccess :=  [Reduction Type: NONE
; CHECK:     { Stmt_for_body[i0] -> MemRef_middle[0] };
; CHECK: ReadAccess :=       [Reduction Type: *
; CHECK:     { Stmt_for_body[i0] -> MemRef_prod[0] };
; CHECK: MustWriteAccess :=  [Reduction Type: *
; CHECK:     { Stmt_for_body[i0] -> MemRef_prod[0] };
; CHECK: ReadAccess :=       [Reduction Type: NONE
; CHECK:     { Stmt_for_body[i0] -> MemRef_A[-1 + i0] };
; CHECK: ReadAccess :=       [Reduction Type: NONE
; CHECK:     { Stmt_for_body[i0] -> MemRef_A[1 + i0] };
; CHECK: MustWriteAccess :=  [Reduction Type: NONE
; CHECK:     { Stmt_for_body[i0] -> MemRef_last[0] };
;
; int first, sum, middle, prod, last;
;
; void f(int * restrict A) {
;   int i;
;   for (int i = 0; i < 100; i++) {
;     first = A[i+1] + A[i];
;     sum += i * 3;
;     middle = A[i-1] + A[i];
;     prod *= (i + 3);
;     last = A[i-1] + A[i+1];
;   }
; }
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

@first = common global i32 0, align 4
@sum = common global i32 0, align 4
@middle = common global i32 0, align 4
@prod = common global i32 0, align 4
@last = common global i32 0, align 4

define void @f(ptr noalias %A) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i1.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %exitcond = icmp ne i32 %i1.0, 100
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %add = add nsw i32 %i1.0, 1
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %add
  %tmp = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %A, i32 %i1.0
  %tmp1 = load i32, ptr %arrayidx2, align 4
  %add3 = add nsw i32 %tmp, %tmp1
  store i32 %add3, ptr @first, align 4
  %mul = mul nsw i32 %i1.0, 3
  %tmp2 = load i32, ptr @sum, align 4
  %add4 = add nsw i32 %tmp2, %mul
  store i32 %add4, ptr @sum, align 4
  %sub = add nsw i32 %i1.0, -1
  %arrayidx5 = getelementptr inbounds i32, ptr %A, i32 %sub
  %tmp3 = load i32, ptr %arrayidx5, align 4
  %arrayidx6 = getelementptr inbounds i32, ptr %A, i32 %i1.0
  %tmp4 = load i32, ptr %arrayidx6, align 4
  %add7 = add nsw i32 %tmp3, %tmp4
  store i32 %add7, ptr @middle, align 4
  %add8 = add nsw i32 %i1.0, 3
  %tmp5 = load i32, ptr @prod, align 4
  %mul9 = mul nsw i32 %tmp5, %add8
  store i32 %mul9, ptr @prod, align 4
  %sub10 = add nsw i32 %i1.0, -1
  %arrayidx11 = getelementptr inbounds i32, ptr %A, i32 %sub10
  %tmp6 = load i32, ptr %arrayidx11, align 4
  %add12 = add nsw i32 %i1.0, 1
  %arrayidx13 = getelementptr inbounds i32, ptr %A, i32 %add12
  %tmp7 = load i32, ptr %arrayidx13, align 4
  %add14 = add nsw i32 %tmp6, %tmp7
  store i32 %add14, ptr @last, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nsw i32 %i1.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
