; RUN: opt %loadNPMPolly -S '-passes=polly<no-default-opts>' \
; RUN:   -plugin-arg=Polly,-polly-vectorize \
; RUN:   -plugin-arg=Polly,-polly-detect-profitability-min-per-loop-insts=0 \
; RUN:   -plugin-arg=Polly,-polly-process-unprofitable=false < %s | FileCheck %s --check-prefix=VECTORIZED
; RUN: opt %loadNPMPolly -S '-passes=polly<no-default-opts>' \
; RUN:   -plugin-arg=Polly,-polly-detect-profitability-min-per-loop-insts=0 \
; RUN:   -plugin-arg=Polly,-polly-process-unprofitable=false < %s | FileCheck %s --check-prefix=NO-VECTORIZE

; Verify that -polly-vectorize bypasses the read-only detection check.
; VECTORIZED-LABEL: define{{.*}} @read_only_sum(
; VECTORIZED: polly.stmt.for.body2:
; VECTORIZED: br {{.*}} !llvm.loop [[POLLY_LOOP:![0-9]+]]
; VECTORIZED-DAG: [[POLLY_LOOP]] = distinct !{[[POLLY_LOOP]], {{.*}}[[ENABLE:![0-9]+]]}
; VECTORIZED-DAG: [[ENABLE]] = !{!"llvm.loop.vectorize.enable"}
;
; NO-VECTORIZE-LABEL: define{{.*}} @read_only_sum(
; NO-VECTORIZE: {{^}}for.body1:
; NO-VECTORIZE-NOT: polly.stmt.

target datalayout = "e-m:e-i8:8:32-i16:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

define i32 @read_only_sum(ptr %A, ptr %B, i64 %n) {
entry:
  %cmp = icmp sgt i64 %n, 0
  br i1 %cmp, label %for.body1, label %for.end

for.body1:
  %i = phi i64 [ 0, %entry ], [ %next, %for.body2 ]
  %sum = phi i32 [ 0, %entry ], [ %sum.next2, %for.body2 ]
  %idx = getelementptr inbounds i32, ptr %A, i64 %i
  %a = load i32, ptr %idx, align 4
  br label %for.body2

for.body2:
  %idx2 = getelementptr inbounds i32, ptr %B, i64 %i
  %b = load i32, ptr %idx2, align 4
  %sum.next = add i32 %sum, %a
  %sum.next2 = add i32 %sum.next, %b
  %next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %next, %n
  br i1 %cond, label %for.body1, label %for.end

for.end:
  %result = phi i32 [ 0, %entry ], [ %sum.next2, %for.body2 ]
  ret i32 %result
}
