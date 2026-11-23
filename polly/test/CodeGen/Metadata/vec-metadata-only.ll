; RUN: opt %loadNPMPolly -S '-passes=polly<no-default-opts>' \
; RUN:   -plugin-arg=Polly,-polly-annotate-metadata-vectorize \
; RUN:   -plugin-arg=Polly,-polly-process-unprofitable=false < %s \
; RUN:   | FileCheck %s

; Verify vectorization metadata is attached to Polly-generated loops, but the
; metadata option does not make an unprofitable loop eligible for vectorization
; CHECK-LABEL: define{{.*}} @nested_loop(
; CHECK: polly.stmt.
; CHECK: br {{.*}} !llvm.loop [[POLLY_LOOP:![0-9]+]]
;
; CHECK-LABEL: define{{.*}} @simple_add(
; CHECK-NOT: polly.stmt.
; CHECK: {{^}}for.body:
; CHECK-NOT: polly.stmt.
;
; CHECK-DAG: [[POLLY_LOOP]] = distinct !{[[POLLY_LOOP]], {{.*}}[[ENABLE:![0-9]+]]}
; CHECK-DAG: [[ENABLE]] = !{!"llvm.loop.vectorize.enable"}

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

define void @nested_loop(ptr %A) {
entry:
  br label %outer

outer:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  br label %inner

inner:
  %j = phi i64 [ 0, %outer ], [ %j.next, %inner ]
  %idx = getelementptr inbounds i32, ptr %A, i64 %j
  %v = load i32, ptr %idx, align 4
  %inc = add nsw i32 %v, 1
  store i32 %inc, ptr %idx, align 4
  %j.next = add nuw nsw i64 %j, 1
  %j.more = icmp slt i64 %j.next, 16
  br i1 %j.more, label %inner, label %outer.latch

outer.latch:
  %i.next = add nuw nsw i64 %i, 1
  %i.more = icmp slt i64 %i.next, 16
  br i1 %i.more, label %outer, label %exit

exit:
  ret void
}

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
