; RUN: opt %loadPolly -pass-remarks-analysis="polly-scops" -polly-scops -disable-output < %s 2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s
;
; REMARK: remark: <unknown>:0:0: Use user assumption: [n, b] -> {  : n <= 100 or (b = 0 and n >= 101) }
;
; CHECK:       Context:
; CHECK-NEXT:    [n, b] -> {  : -9223372036854775808 <= n <= 9223372036854775807 and ((n <= 100 and -9223372036854775808 <= b <= 9223372036854775807) or (b = 0 and n >= 101)) }
; CHECK-NEXT:    Assumed Context:
; CHECK-NEXT:    [n, b] -> {  :  }
; CHECK-NEXT:    Invalid Context:
; CHECK-NEXT:    [n, b] -> {  : 1 = 0 }

;
;    void foo(float A[][100], long b, long n) {
;      for (long i = 0; i < n; i++)
;        if (b)
;          A[42][i] = 42.0;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@.src = private unnamed_addr constant [12 x i8] c"/tmp/test.c\00", align 1
@0 = private unnamed_addr constant { i16, i16, [14 x i8] } { i16 -1, i16 0, [14 x i8] c"'float [100]'\00" }
@1 = private unnamed_addr constant { i16, i16, [7 x i8] } { i16 0, i16 13, [7 x i8] c"'long'\00" }

define void @foo([100 x float]* %A, i64 %b, i64 %n) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i64 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i64 %i.0, %n
  br i1 %cmp, label %if.cond, label %for.end

if.cond:
  %bcmp = icmp ne i64 %b, 0
  br i1 %bcmp, label %for.body, label %for.inc

for.body:                                         ; preds = %for.cond
  %tmp = icmp slt i64 %i.0, 100
  call void @llvm.assume(i1 %tmp)
  %arrayidx1 = getelementptr inbounds [100 x float], [100 x float]* %A, i64 42, i64 %i.0
  store float 4.200000e+01, float* %arrayidx1, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nuw nsw i64 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

declare void @llvm.assume(i1) #1

