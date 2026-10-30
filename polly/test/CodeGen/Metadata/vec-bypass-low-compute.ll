; RUN: opt %loadNPMPolly -S '-passes=polly<no-default-opts>' \
; RUN:   -plugin-arg=Polly,-polly-vectorize \
; RUN:   -plugin-arg=Polly,-polly-process-unprofitable=false < %s | FileCheck %s --check-prefix=VECTORIZED
; RUN: opt %loadNPMPolly -S '-passes=polly<no-default-opts>' \
; RUN:   -plugin-arg=Polly,-polly-process-unprofitable=false < %s | FileCheck %s --check-prefix=NO-VECTORIZE

; Verify that -polly-vectorize bypasses the low-compute detection check.
; VECTORIZED-LABEL: define{{.*}} @simple_add(
; VECTORIZED: polly.stmt.for.body:
; VECTORIZED: br {{.*}} !llvm.loop [[POLLY_LOOP:![0-9]+]]
; VECTORIZED-DAG: [[POLLY_LOOP]] = distinct !{[[POLLY_LOOP]], {{.*}}[[ENABLE:![0-9]+]]}
; VECTORIZED-DAG: [[ENABLE]] = !{!"llvm.loop.vectorize.enable"}
;
; NO-VECTORIZE-LABEL: define{{.*}} @simple_add(
; NO-VECTORIZE: {{^}}for.body:
; NO-VECTORIZE-NOT: polly.stmt.

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

define void @simple_add(ptr noalias %A, ptr noalias %B, i64 %n) {
entry:
  %cmp = icmp sgt i64 %n, 0
  br i1 %cmp, label %for.body, label %for.end

for.body:
  %i = phi i64 [ 0, %entry ], [ %next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %i
  %v = load i32, ptr %arrayidx, align 4
  %inc = add nsw i32 %v, 1
  store i32 %inc, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %B, i64 %i
  store i32 1, ptr %arrayidx2, align 4
  %next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %next, %n
  br i1 %cond, label %for.body, label %for.end, !llvm.loop !0

for.end:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.mustprogress"}
