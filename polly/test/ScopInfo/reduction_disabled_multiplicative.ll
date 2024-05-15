; RUN: opt -basic-aa %loadPolly -polly-stmt-granularity=bb -polly-print-scops -polly-disable-multiplicative-reductions -disable-output < %s | FileCheck %s
;
; CHECK: ReadAccess :=       [Reduction Type: +
; CHECK:     { Stmt_for_body[i0] -> MemRef_sum[0] };
; CHECK: MustWriteAccess :=  [Reduction Type: +
; CHECK:     { Stmt_for_body[i0] -> MemRef_sum[0] };
; CHECK: ReadAccess :=       [Reduction Type: NONE
; CHECK:     { Stmt_for_body[i0] -> MemRef_prod[0] };
; CHECK: MustWriteAccess :=  [Reduction Type: NONE
; CHECK:     { Stmt_for_body[i0] -> MemRef_prod[0] };
;
; int sum, prod;
;
; void f() {
;   int i;
;   for (int i = 0; i < 100; i++) {
;     sum += i * 3;
;     prod *= (i + 3);
;   }
; }
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

@sum = common global i32 0, align 4
@prod = common global i32 0, align 4

define void @f() #0 {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i1.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %exitcond = icmp ne i32 %i1.0, 100
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %mul = mul nsw i32 %i1.0, 3
  %tmp = load i32, ptr @sum, align 4
  %add = add nsw i32 %tmp, %mul
  store i32 %add, ptr @sum, align 4
  %add2 = add nsw i32 %i1.0, 3
  %tmp1 = load i32, ptr @prod, align 4
  %mul3 = mul nsw i32 %tmp1, %add2
  store i32 %mul3, ptr @prod, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nsw i32 %i1.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
