; Positive tests for -allow-unj-safe-intrinsics.
;
; Each function is a jammable nest (B is write-only at B[i*M+j], A/B noalias, so
; there is no cross-iteration dependence) whose ONLY obstacle is an intrinsic
; that nominally reports mayReadOrWriteMemory() and therefore makes the
; conservative bailout in getLoadsAndStores() reject the loop.
;
; These intrinsics impose no real dependence, so the relaxation should admit
; them: each nest must stay un-jammed with the relaxation disabled (BASELINE)
; and become jammed by default (TRANSFORM). The paired file
; safe-intrinsics-negative.ll checks the other direction.
;
; RUN: opt -passes='loop-unroll-and-jam' -allow-unroll-and-jam \
; RUN:     -unroll-and-jam-count=2 -allow-unj-safe-intrinsics=false < %s -S \
; RUN:   | FileCheck %s --check-prefix=BASELINE
;
; RUN: opt -passes='loop-unroll-and-jam' -allow-unroll-and-jam \
; RUN:     -unroll-and-jam-count=2 < %s -S \
; RUN:   | FileCheck %s --check-prefix=TRANSFORM

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

declare void @llvm.assume(i1)
declare void @llvm.lifetime.start.p0(i64, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64, ptr nocapture)
declare ptr @llvm.invariant.start.p0(i64, ptr nocapture)
declare void @llvm.invariant.end.p0(ptr, i64, ptr nocapture)

;------------------------------------------------------------------------------
; P1: lifetime markers around an alloca. These are argmem-only stack markers;
; the substantive accesses to that alloca stay in the dependence set, so they
; are admitted explicitly by the predicate.
; BASELINE-LABEL: @lifetime_pos(
; BASELINE-NOT: add nuw nsw i64 %i, 2
; TRANSFORM-LABEL: @lifetime_pos(
; TRANSFORM: add nuw nsw i64 %i, 2
define void @lifetime_pos(ptr noalias %A, ptr noalias %B, i64 %N, i64 %M) {
entry:
  %scratch = alloca float
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  br label %inner.preheader
inner.preheader:
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %inner.preheader ], [ %j.next, %inner.header ]
  call void @llvm.lifetime.start.p0(i64 4, ptr %scratch)
  %a.idx = getelementptr inbounds float, ptr %A, i64 %j
  %a.val = load float, ptr %a.idx, align 4
  %row = mul i64 %i, %M
  %idx = add i64 %row, %j
  %b.ptr = getelementptr inbounds float, ptr %B, i64 %idx
  store float %a.val, ptr %b.ptr, align 4
  call void @llvm.lifetime.end.p0(i64 4, ptr %scratch)
  %j.next = add nuw nsw i64 %j, 1
  %inner.cond = icmp slt i64 %j.next, %M
  br i1 %inner.cond, label %inner.header, label %outer.latch
outer.latch:
  %i.next = add nuw nsw i64 %i, 1
  %outer.cond = icmp slt i64 %i.next, %N
  br i1 %outer.cond, label %outer.header, label %outer.exit
outer.exit:
  ret void
}

;------------------------------------------------------------------------------
; P2: llvm.assume. Reports memory(inaccessiblemem: write), so it is admitted by
; the GENERIC onlyAccessesInaccessibleMemory() rule rather than by name --
; inaccessible memory cannot alias the loop's loads/stores.
; BASELINE-LABEL: @assume_pos(
; BASELINE-NOT: add nuw nsw i64 %i, 2
; TRANSFORM-LABEL: @assume_pos(
; TRANSFORM: add nuw nsw i64 %i, 2
define void @assume_pos(ptr noalias %A, ptr noalias %B, i64 %N, i64 %M) {
entry:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  br label %inner.preheader
inner.preheader:
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %inner.preheader ], [ %j.next, %inner.header ]
  %a.idx = getelementptr inbounds float, ptr %A, i64 %j
  %a.val = load float, ptr %a.idx, align 4
  %cond = fcmp ogt float %a.val, -1.000000e+30
  call void @llvm.assume(i1 %cond)
  %row = mul i64 %i, %M
  %idx = add i64 %row, %j
  %b.ptr = getelementptr inbounds float, ptr %B, i64 %idx
  store float %a.val, ptr %b.ptr, align 4
  %j.next = add nuw nsw i64 %j, 1
  %inner.cond = icmp slt i64 %j.next, %M
  br i1 %inner.cond, label %inner.header, label %outer.latch
outer.latch:
  %i.next = add nuw nsw i64 %i, 1
  %outer.cond = icmp slt i64 %i.next, %N
  br i1 %outer.cond, label %outer.header, label %outer.exit
outer.exit:
  ret void
}

;------------------------------------------------------------------------------
; P3: invariant.start/end markers -- argmem-only scope markers, admitted
; explicitly for the same reason as the lifetime markers.
; BASELINE-LABEL: @invariant_pos(
; BASELINE-NOT: add nuw nsw i64 %i, 2
; TRANSFORM-LABEL: @invariant_pos(
; TRANSFORM: add nuw nsw i64 %i, 2
define void @invariant_pos(ptr noalias %A, ptr noalias %B, i64 %N, i64 %M) {
entry:
  %scratch = alloca float
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  br label %inner.preheader
inner.preheader:
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %inner.preheader ], [ %j.next, %inner.header ]
  %inv = call ptr @llvm.invariant.start.p0(i64 4, ptr %scratch)
  %a.idx = getelementptr inbounds float, ptr %A, i64 %j
  %a.val = load float, ptr %a.idx, align 4
  %row = mul i64 %i, %M
  %idx = add i64 %row, %j
  %b.ptr = getelementptr inbounds float, ptr %B, i64 %idx
  store float %a.val, ptr %b.ptr, align 4
  call void @llvm.invariant.end.p0(ptr %inv, i64 4, ptr %scratch)
  %j.next = add nuw nsw i64 %j, 1
  %inner.cond = icmp slt i64 %j.next, %M
  br i1 %inner.cond, label %inner.header, label %outer.latch
outer.latch:
  %i.next = add nuw nsw i64 %i, 1
  %outer.cond = icmp slt i64 %i.next, %N
  br i1 %outer.cond, label %outer.header, label %outer.exit
outer.exit:
  ret void
}

;------------------------------------------------------------------------------
; P4: lifetime markers AND assume together -- exercises both the explicit
; argmem-marker path and the generic inaccessible-memory path in one nest.
; BASELINE-LABEL: @combined_pos(
; BASELINE-NOT: add nuw nsw i64 %i, 2
; TRANSFORM-LABEL: @combined_pos(
; TRANSFORM: add nuw nsw i64 %i, 2
define void @combined_pos(ptr noalias %A, ptr noalias %B, i64 %N, i64 %M) {
entry:
  %scratch = alloca float
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  br label %inner.preheader
inner.preheader:
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %inner.preheader ], [ %j.next, %inner.header ]
  call void @llvm.lifetime.start.p0(i64 4, ptr %scratch)
  %a.idx = getelementptr inbounds float, ptr %A, i64 %j
  %a.val = load float, ptr %a.idx, align 4
  %cond = fcmp ogt float %a.val, -1.000000e+30
  call void @llvm.assume(i1 %cond)
  %row = mul i64 %i, %M
  %idx = add i64 %row, %j
  %b.ptr = getelementptr inbounds float, ptr %B, i64 %idx
  store float %a.val, ptr %b.ptr, align 4
  call void @llvm.lifetime.end.p0(i64 4, ptr %scratch)
  %j.next = add nuw nsw i64 %j, 1
  %inner.cond = icmp slt i64 %j.next, %M
  br i1 %inner.cond, label %inner.header, label %outer.latch
outer.latch:
  %i.next = add nuw nsw i64 %i, 1
  %outer.cond = icmp slt i64 %i.next, %N
  br i1 %outer.cond, label %outer.header, label %outer.exit
outer.exit:
  ret void
}
