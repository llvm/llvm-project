; Collapse only the prefix of independent, constant-bound loops needed to
; supply the requested OpenMP team. A zero thread count defers to the runtime.
;
; RUN: opt %loadNPMPolly '-passes=polly-custom<codegen>' -plugin-arg=Polly,-polly-process-unprofitable -plugin-arg=Polly,-polly-parallel -plugin-arg=Polly,-polly-omp-backend=LLVM -plugin-arg=Polly,-polly-num-threads=0 -verify-each -S < %s | FileCheck %s --check-prefixes=COMMON,SMALL
; RUN: opt %loadNPMPolly '-passes=polly-custom<codegen>' -plugin-arg=Polly,-polly-process-unprofitable -plugin-arg=Polly,-polly-parallel -plugin-arg=Polly,-polly-omp-backend=LLVM -plugin-arg=Polly,-polly-num-threads=1 -verify-each -S < %s | FileCheck %s --check-prefixes=COMMON,SMALL
; RUN: opt %loadNPMPolly '-passes=polly-custom<codegen>' -plugin-arg=Polly,-polly-process-unprofitable -plugin-arg=Polly,-polly-parallel -plugin-arg=Polly,-polly-omp-backend=LLVM -plugin-arg=Polly,-polly-num-threads=4 -verify-each -S < %s | FileCheck %s --check-prefixes=COMMON,SMALL
; RUN: opt %loadNPMPolly '-passes=polly-custom<codegen>' -plugin-arg=Polly,-polly-process-unprofitable -plugin-arg=Polly,-polly-parallel -plugin-arg=Polly,-polly-omp-backend=LLVM -plugin-arg=Polly,-polly-num-threads=16 -verify-each -S < %s | FileCheck %s --check-prefixes=COMMON,MEDIUM
; RUN: opt %loadNPMPolly '-passes=polly-custom<codegen>' -plugin-arg=Polly,-polly-process-unprofitable -plugin-arg=Polly,-polly-parallel -plugin-arg=Polly,-polly-omp-backend=LLVM -plugin-arg=Polly,-polly-num-threads=64 -verify-each -S < %s | FileCheck %s --check-prefixes=COMMON,WIDE
; RUN: opt %loadNPMPolly '-passes=polly-custom<codegen>' -plugin-arg=Polly,-polly-process-unprofitable -plugin-arg=Polly,-polly-parallel -plugin-arg=Polly,-polly-omp-backend=LLVM -plugin-arg=Polly,-polly-num-threads=256 -verify-each -S < %s | FileCheck %s --check-prefixes=COMMON,WIDE

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; A 4 x 4 x 8 independent nest stops at 4, 16, or 128 work units.
; COMMON-LABEL: define {{.*}}@cube(
; SMALL: call void {{.*}}@__kmpc_fork_call({{.*}}i64 0, i64 4, i64 1,
; MEDIUM: call void {{.*}}@__kmpc_fork_call({{.*}}i64 0, i64 16, i64 1,
; WIDE: call void {{.*}}@__kmpc_fork_call({{.*}}i64 0, i64 128, i64 1,
define void @cube(ptr noalias %out, ptr noalias %in) {
entry:
  br label %i.header

i.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %i.latch ]
  %i.in.bounds = icmp slt i64 %i, 4
  br i1 %i.in.bounds, label %j.header, label %exit

j.header:
  %j = phi i64 [ 0, %i.header ], [ %j.next, %j.latch ]
  %j.in.bounds = icmp slt i64 %j, 4
  br i1 %j.in.bounds, label %k.header, label %i.latch

k.header:
  %k = phi i64 [ 0, %j.header ], [ %k.next, %body ]
  %k.in.bounds = icmp slt i64 %k, 8
  br i1 %k.in.bounds, label %body, label %j.latch

body:
  %src = getelementptr inbounds [4 x [8 x i32]], ptr %in, i64 %i, i64 %j, i64 %k
  %dst = getelementptr inbounds [4 x [8 x i32]], ptr %out, i64 %i, i64 %j, i64 %k
  %input = load i32, ptr %src, align 4
  %result = add nsw i32 %input, 3
  store i32 %result, ptr %dst, align 4
  %k.next = add nsw i64 %k, 1
  br label %k.header

j.latch:
  %j.next = add nsw i64 %j, 1
  br label %j.header

i.latch:
  %i.next = add nsw i64 %i, 1
  br label %i.header

exit:
  ret void
}

; Stop at the dependent middle loop, even though the inner loop is parallel.
; COMMON-LABEL: define {{.*}}@dependent_middle(
; COMMON: call void {{.*}}@__kmpc_fork_call({{.*}}i64 0, i64 4, i64 1,
define void @dependent_middle(ptr noalias %out, ptr noalias %in) {
entry:
  br label %i.header

i.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %i.latch ]
  %i.in.bounds = icmp slt i64 %i, 4
  br i1 %i.in.bounds, label %j.header, label %exit

j.header:
  %j = phi i64 [ 1, %i.header ], [ %j.next, %j.latch ]
  %j.in.bounds = icmp slt i64 %j, 4
  br i1 %j.in.bounds, label %k.header, label %i.latch

k.header:
  %k = phi i64 [ 0, %j.header ], [ %k.next, %body ]
  %k.in.bounds = icmp slt i64 %k, 8
  br i1 %k.in.bounds, label %body, label %j.latch

body:
  %previous.j = sub nsw i64 %j, 1
  %previous = getelementptr inbounds [4 x [8 x i32]], ptr %out, i64 %i, i64 %previous.j, i64 %k
  %src = getelementptr inbounds i32, ptr %in, i64 %k
  %dst = getelementptr inbounds [4 x [8 x i32]], ptr %out, i64 %i, i64 %j, i64 %k
  %carried = load i32, ptr %previous, align 4
  %input = load i32, ptr %src, align 4
  %result = add nsw i32 %carried, %input
  store i32 %result, ptr %dst, align 4
  %k.next = add nsw i64 %k, 1
  br label %k.header

j.latch:
  %j.next = add nsw i64 %j, 1
  br label %j.header

i.latch:
  %i.next = add nsw i64 %i, 1
  br label %i.header

exit:
  ret void
}

; A reduction-parallel loop also terminates the collapsed prefix.
; COMMON-LABEL: define {{.*}}@reduction_middle(
; COMMON: call void {{.*}}@__kmpc_fork_call({{.*}}i64 0, i64 4, i64 1,
define void @reduction_middle(ptr noalias %out, ptr noalias %in) {
entry:
  br label %i.header

i.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %i.latch ]
  %i.in.bounds = icmp slt i64 %i, 4
  br i1 %i.in.bounds, label %j.header, label %exit

j.header:
  %j = phi i64 [ 0, %i.header ], [ %j.next, %j.latch ]
  %j.in.bounds = icmp slt i64 %j, 4
  br i1 %j.in.bounds, label %k.header, label %i.latch

k.header:
  %k = phi i64 [ 0, %j.header ], [ %k.next, %body ]
  %k.in.bounds = icmp slt i64 %k, 8
  br i1 %k.in.bounds, label %body, label %j.latch

body:
  %src = getelementptr inbounds [8 x i32], ptr %in, i64 %j, i64 %k
  %dst = getelementptr inbounds [8 x i32], ptr %out, i64 %i, i64 %k
  %input = load i32, ptr %src, align 4
  %accumulator = load i32, ptr %dst, align 4
  %sum = add nsw i32 %accumulator, %input
  store i32 %sum, ptr %dst, align 4
  %k.next = add nsw i64 %k, 1
  br label %k.header

j.latch:
  %j.next = add nsw i64 %j, 1
  br label %j.header

i.latch:
  %i.next = add nsw i64 %i, 1
  br label %i.header

exit:
  ret void
}

; Preserve statements between loops.
; COMMON-LABEL: define {{.*}}@imperfect(
; COMMON: call void {{.*}}@__kmpc_fork_call({{.*}}i64 0, i64 4, i64 1,
define void @imperfect(ptr noalias %out, ptr noalias %in) {
entry:
  br label %i.header

i.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %i.latch ]
  %i.in.bounds = icmp slt i64 %i, 4
  br i1 %i.in.bounds, label %before.j, label %exit

before.j:
  %extra.index = add nsw i64 %i, 64
  %extra.src = getelementptr inbounds i32, ptr %in, i64 %i
  %extra.dst = getelementptr inbounds i32, ptr %out, i64 %extra.index
  %extra.value = load i32, ptr %extra.src, align 4
  store i32 %extra.value, ptr %extra.dst, align 4
  br label %j.header

j.header:
  %j = phi i64 [ 0, %before.j ], [ %j.next, %body ]
  %j.in.bounds = icmp slt i64 %j, 4
  br i1 %j.in.bounds, label %body, label %i.latch

body:
  %src = getelementptr inbounds [4 x i32], ptr %in, i64 %i, i64 %j
  %dst = getelementptr inbounds [4 x i32], ptr %out, i64 %i, i64 %j
  %input = load i32, ptr %src, align 4
  %result = add nsw i32 %input, 7
  store i32 %result, ptr %dst, align 4
  %j.next = add nsw i64 %j, 1
  br label %j.header

i.latch:
  %i.next = add nsw i64 %i, 1
  br label %i.header

exit:
  ret void
}

; A runtime-dependent outer trip count retains its original bounds.
; COMMON-LABEL: define {{.*}}@dynamic(
; COMMON: call void {{.*}}@__kmpc_fork_call({{.*}}i64 0, i64 %{{[^,]+}}, i64 1,
define void @dynamic(ptr noalias %out, ptr noalias %in, i64 %n) {
entry:
  br label %i.header

i.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %i.latch ]
  %i.in.bounds = icmp slt i64 %i, %n
  br i1 %i.in.bounds, label %j.header, label %exit

j.header:
  %j = phi i64 [ 0, %i.header ], [ %j.next, %body ]
  %j.in.bounds = icmp slt i64 %j, 4
  br i1 %j.in.bounds, label %body, label %i.latch

body:
  %src = getelementptr inbounds [4 x i32], ptr %in, i64 %i, i64 %j
  %dst = getelementptr inbounds [4 x i32], ptr %out, i64 %i, i64 %j
  %input = load i32, ptr %src, align 4
  %result = add nsw i32 %input, 11
  store i32 %result, ptr %dst, align 4
  %j.next = add nsw i64 %j, 1
  br label %j.header

i.latch:
  %i.next = add nsw i64 %i, 1
  br label %i.header

exit:
  ret void
}

; Nonzero source lower bounds and strides preserve their iteration mapping.
; COMMON-LABEL: define {{.*}}@strided(
; SMALL: call void {{.*}}@__kmpc_fork_call({{.*}}i64 0, i64 4, i64 1,
; MEDIUM: call void {{.*}}@__kmpc_fork_call({{.*}}i64 0, i64 28, i64 1,
; WIDE: call void {{.*}}@__kmpc_fork_call({{.*}}i64 0, i64 28, i64 1,
define void @strided(ptr noalias %out, ptr noalias %in) {
entry:
  br label %i.header

i.header:
  %i = phi i64 [ -3, %entry ], [ %i.next, %i.latch ]
  %i.in.bounds = icmp slt i64 %i, 5
  br i1 %i.in.bounds, label %j.header, label %exit

j.header:
  %j = phi i64 [ 7, %i.header ], [ %j.next, %body ]
  %j.in.bounds = icmp slt i64 %j, 26
  br i1 %j.in.bounds, label %body, label %i.latch

body:
  %row = add nsw i64 %i, 3
  %src = getelementptr inbounds [32 x i32], ptr %in, i64 %row, i64 %j
  %dst = getelementptr inbounds [32 x i32], ptr %out, i64 %row, i64 %j
  %input = load i32, ptr %src, align 4
  %result = add nsw i32 %input, 13
  store i32 %result, ptr %dst, align 4
  %j.next = add nsw i64 %j, 3
  br label %j.header

i.latch:
  %i.next = add nsw i64 %i, 2
  br label %i.header

exit:
  ret void
}

; Retain a collapsed 4 x 4 prefix when a deeper loop carries dependences.
; COMMON-LABEL: define {{.*}}@partial(
; SMALL: call void {{.*}}@__kmpc_fork_call({{.*}}i64 0, i64 4, i64 1,
; MEDIUM: call void {{.*}}@__kmpc_fork_call({{.*}}i64 0, i64 16, i64 1,
; WIDE: call void {{.*}}@__kmpc_fork_call({{.*}}i64 0, i64 16, i64 1,
define void @partial(ptr noalias %out, ptr noalias %in) {
entry:
  br label %i.header

i.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %i.latch ]
  %i.in.bounds = icmp slt i64 %i, 4
  br i1 %i.in.bounds, label %j.header, label %exit

j.header:
  %j = phi i64 [ 0, %i.header ], [ %j.next, %j.latch ]
  %j.in.bounds = icmp slt i64 %j, 4
  br i1 %j.in.bounds, label %k.header, label %i.latch

k.header:
  %k = phi i64 [ 1, %j.header ], [ %k.next, %body ]
  %k.in.bounds = icmp slt i64 %k, 8
  br i1 %k.in.bounds, label %body, label %j.latch

body:
  %previous.k = sub nsw i64 %k, 1
  %previous = getelementptr inbounds [4 x [8 x i32]], ptr %out, i64 %i, i64 %j, i64 %previous.k
  %src = getelementptr inbounds i32, ptr %in, i64 %k
  %dst = getelementptr inbounds [4 x [8 x i32]], ptr %out, i64 %i, i64 %j, i64 %k
  %carried = load i32, ptr %previous, align 4
  %input = load i32, ptr %src, align 4
  %result = add nsw i32 %carried, %input
  store i32 %result, ptr %dst, align 4
  %k.next = add nsw i64 %k, 1
  br label %k.header

j.latch:
  %j.next = add nsw i64 %j, 1
  br label %j.header

i.latch:
  %i.next = add nsw i64 %i, 1
  br label %i.header

exit:
  ret void
}
