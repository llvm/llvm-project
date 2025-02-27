; RUN: opt %loadNPMPolly -passes=polly-codegen \
; RUN:     -S < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; CHECK: sext i32 %indvar.init to i64

define void @foo(ptr %A, ptr %B, i32 %p, i32 %indvar.init) {
preheader:
  br label %for.body

for.body:
  %indvar = phi i32 [ %indvar.next, %for.body ], [ %indvar.init, %preheader ]
  %tmp1 = load double, ptr %B
  store double undef, ptr %A
  %idxprom1329 = sext i32 %indvar to i64
  %B.ptr1 = getelementptr inbounds double, ptr %B, i64 %idxprom1329
  store double 0.000000e+00, ptr %B.ptr1
  %indvar.next = add nsw i32 %indvar, %p
  br i1 false, label %for.body, label %exit

exit:
  ret void
}
