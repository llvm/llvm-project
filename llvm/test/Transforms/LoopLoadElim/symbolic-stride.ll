; RUN: opt -loop-load-elim -S < %s | FileCheck %s

; Forwarding in the presence of symbolic strides is currently not supported:
;
;   for (unsigned i = 0; i < 100; i++)
;     A[i + 1] = A[Stride * i] + B[i];

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

; CHECK-LABEL: @f(
define void @f(i32* noalias nocapture %A, i32* noalias nocapture readonly %B, i64 %N,
               i64 %stride) {
entry:
; CHECK-NOT: %load_initial = load i32, i32* %A
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
; CHECK-NOT: %store_forwarded = phi i32 [ %load_initial, {{.*}} ], [ %add, %for.body ]
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %mul = mul i64 %indvars.iv, %stride
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %mul
  %load = load i32, i32* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %B, i64 %indvars.iv
  %load_1 = load i32, i32* %arrayidx2, align 4
; CHECK-NOT: %add = add i32 %load_1, %store_forwarded
  %add = add i32 %load_1, %load
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %arrayidx_next = getelementptr inbounds i32, i32* %A, i64 %indvars.iv.next
  store i32 %add, i32* %arrayidx_next, align 4
  %exitcond = icmp eq i64 %indvars.iv.next, %N
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}
