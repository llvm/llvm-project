; RUN: opt %loadNPMPolly '-passes=print<polly-detect>' -disable-output < %s 2>&1 | FileCheck %s

;  foo(float *A, long n, long k) {
;    if (true)
;      A[n * k] = 0;
;  }
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

define void @foo(ptr %A, i64 %n, i64 %k) {
entry:
  br label %for.j

for.j:
  br i1 true, label %if.then, label %return

if.then:
  %mul = mul nsw i64 %n, %k
  %arrayidx = getelementptr float, ptr %A, i64 %mul
  store float 0.000000e+00, ptr %arrayidx
  br label %return

return:
  ret void
}

; CHECK: Valid Region for Scop: for.j => return
