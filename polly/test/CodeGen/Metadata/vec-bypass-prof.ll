; RUN: opt %loadNPMPolly -S '-passes=polly<no-default-opts>' \
; RUN:   -polly-annotate-metadata-vectorize -polly-process-unprofitable=false -plugin-arg=Polly,-polly-process-unprofitable=false < %s | FileCheck %s --check-prefix=VECTORIZED
; RUN: opt %loadNPMPolly -S '-passes=polly<no-default-opts>' \
; RUN:   -polly-process-unprofitable=false -plugin-arg=Polly,-polly-process-unprofitable=false < %s | FileCheck %s --check-prefix=NO-VECTORIZE

; A simple single-level loop is unprofitable without -polly-annotate-metadata-vectorize
;
; VECTORIZED-LABEL: define{{.*}} @simple_add(
; VECTORIZED-DAG: polly.stmt.for.body:
; VECTORIZED-DAG: br {{.*}} !llvm.loop [[SIMPLE_LOOP:![0-9]+]]
; VECTORIZED-DAG: [[SIMPLE_LOOP]] = distinct !{[[SIMPLE_LOOP]],
; VECTORIZED-DAG: !{!"llvm.loop.vectorize.enable"}
;
; NO-VECTORIZE-LABEL: define{{.*}} @simple_add(
; NO-VECTORIZE: for.body:
; NO-VECTORIZE-NOT: polly.stmt.for.body:

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

define void @simple_add(ptr %A, i64 %n) {
entry:
  %cmp = icmp sgt i64 %n, 0
  br i1 %cmp, label %for.body, label %for.end

for.body:
  %i = phi i64 [ 0, %entry ], [ %next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %i
  %v = load i32, ptr %arrayidx, align 4
  %inc = add nsw i32 %v, 1
  store i32 %inc, ptr %arrayidx, align 4
  %next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %next, %n
  br i1 %cond, label %for.body, label %for.end, !llvm.loop !0

for.end:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.mustprogress"}
