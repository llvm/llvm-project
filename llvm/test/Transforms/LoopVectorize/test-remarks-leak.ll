; RUN: opt < %s -passes=loop-vectorize -force-vector-width=4 -S -pass-remarks-missed='loop-vectorize' 2>&1 | FileCheck %s -check-prefix=CHECK-MISSED-OPT-REMARK
; RUN: opt < %s -passes=loop-vectorize -force-vector-width=4 -S -pass-remarks='loop-vectorize' 2>&1 | FileCheck %s -check-prefix=CHECK-REMARK

; Test that only specified remarks get emitted

; CHECK-MISSED-OPT-REMARK: remark: {{.*}} loop not vectorized

; NOTE: -pass-remarks should only report remarks for successful vectorization
; CHECK-REMARK-NOT: remark: {{.*}} loop not vectorized: Cannot determine whether critical uncountable exit load address does not alias with a memory write


target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

define i32 @_Z4testPii(ptr nocapture %A, i32 %Length) {
entry:
  %cmp8 = icmp sgt i32 %Length, 0
  br i1 %cmp8, label %for.body.preheader, label %end

for.body.preheader:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %if.else ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %cmp1 = icmp sgt i32 %0, 10
  br i1 %cmp1, label %end.loopexit, label %if.else

if.else:
  store i32 0, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %1 = trunc i64 %indvars.iv.next to i32
  %cmp = icmp slt i32 %1, %Length
  br i1 %cmp, label %for.body, label %end.loopexit

end.loopexit:
  br label %end

end:
  ret i32 0
}
