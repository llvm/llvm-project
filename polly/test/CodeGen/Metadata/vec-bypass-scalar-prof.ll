; RUN: opt %loadNPMPolly -S '-passes=polly<no-default-opts>' \
; RUN:   -plugin-arg=Polly,-polly-vectorize \
; RUN:   -plugin-arg=Polly,-polly-detect-profitability-min-per-loop-insts=0 \
; RUN:   -plugin-arg=Polly,-polly-unprofitable-scalar-accs=true \
; RUN:   -plugin-arg=Polly,-polly-enable-prune-unprofitable=false \
; RUN:   -plugin-arg=Polly,-polly-process-unprofitable=false \
; RUN:   < %s | FileCheck %s --check-prefix=VECTORIZED
; RUN: opt %loadNPMPolly -S '-passes=polly<no-default-opts>' \
; RUN:   -plugin-arg=Polly,-polly-detect-profitability-min-per-loop-insts=0 \
; RUN:   -plugin-arg=Polly,-polly-unprofitable-scalar-accs=true \
; RUN:   -plugin-arg=Polly,-polly-enable-prune-unprofitable=false \
; RUN:   -plugin-arg=Polly,-polly-process-unprofitable=false \
; RUN:   < %s | FileCheck %s --check-prefix=NO-VECTORIZE

; Verify that -polly-vectorize bypasses the scalar-access profitability check.
; VECTORIZED-LABEL: define{{.*}} @scalar_loop(
; VECTORIZED: polly.stmt.for.body:
; VECTORIZED: polly.stmt.for.body1:
; VECTORIZED: br {{.*}} !llvm.loop [[POLLY_LOOP:![0-9]+]]
; VECTORIZED-DAG: [[POLLY_LOOP]] = distinct !{[[POLLY_LOOP]], {{.*}}[[ENABLE:![0-9]+]]}
; VECTORIZED-DAG: [[ENABLE]] = !{!"llvm.loop.vectorize.enable"}
;
; NO-VECTORIZE-LABEL: define{{.*}} @scalar_loop(
; NO-VECTORIZE: {{^}}for.body:
; NO-VECTORIZE-NOT: polly.stmt.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @scalar_loop(ptr noalias %A, ptr noalias %B, i64 %n) {
entry:
  %cmp = icmp sgt i64 %n, 0
  br i1 %cmp, label %for.body, label %for.end

for.body:
  %i = phi i64 [ 0, %entry ], [ %next, %for.body ]
  %x = phi i32 [ 0, %entry ], [ %y, %for.body ]
  %idx = getelementptr inbounds i32, ptr %A, i64 %i
  %v = load i32, ptr %idx, align 4
  %y = add i32 %x, %v
  store i32 %y, ptr %idx, align 4
  %idx2 = getelementptr inbounds i32, ptr %B, i64 %i
  store i32 1, ptr %idx2, align 4
  %next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}
