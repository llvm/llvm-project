; RUN: opt %loadNPMPolly -pass-remarks-analysis="polly-scops" '-passes=print<polly-function-scops>' -disable-output < %s 2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -disable-output < %s 2>&1 | FileCheck %s
;
; REMARK:      remark: <unknown>:0:0: SCoP begins here.
; REMARK-NEXT: remark: <unknown>:0:0: Use user assumption: [n] -> {  : n <= 100 }
; REMARK-NEXT: remark: <unknown>:0:0: Signed-unsigned restriction: [n] -> {  : n < 0 }
; REMARK-NEXT: remark: <unknown>:0:0: SCoP ends here.
;
; CHECK: Context:
; CHECK-NEXT: [n] -> {  : -9223372036854775808 <= n <= 100 }
;
;    void foo(float A[][100], long n) {
;      for (long i = 0; i < n; i++)
;        A[42][i] = 42.0;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@.src = private unnamed_addr constant [12 x i8] c"/tmp/test.c\00", align 1
@0 = private unnamed_addr constant { i16, i16, [14 x i8] } { i16 -1, i16 0, [14 x i8] c"'float [100]'\00" }
@1 = private unnamed_addr constant { i16, i16, [7 x i8] } { i16 0, i16 13, [7 x i8] c"'long'\00" }

define void @foo(ptr %A, i64 %n) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i64 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp ult i64 %i.0, %n
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %tmp = icmp slt i64 %i.0, 100
  call void @llvm.assume(i1 %tmp)
  %arrayidx1 = getelementptr inbounds [100 x float], ptr %A, i64 42, i64 %i.0
  store float 4.200000e+01, ptr %arrayidx1, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nuw nsw i64 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

declare void @llvm.assume(i1) #1

