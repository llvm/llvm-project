; Negative tests for -allow-unj-safe-intrinsics.
;
; Each function is an otherwise-jammable nest (write-only B, no real dep) that
; contains ONE construct which must STILL block unroll-and-jam even when the
; safe-intrinsics relaxation is enabled -- because it either (a) touches real,
; alias-visible memory, (b) is a non-simple (volatile/atomic) access, (c) needs
; metadata cloning we don't perform, (d) is an opaque call, or (e) coexists with
; a genuine cross-iteration dependence.
;
; If any of these DID get jammed, the relaxation would be a miscompile.
;
; RUN: opt -passes='loop-unroll-and-jam' -allow-unroll-and-jam \
; RUN:     -unroll-and-jam-count=2 < %s -S \
; RUN:   | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)
declare void @llvm.masked.store.v4f32.p0(<4 x float>, ptr, i32, <4 x i1>)
declare void @llvm.assume(i1)
declare void @llvm.experimental.noalias.scope.decl(metadata)
declare void @ext_func()

;------------------------------------------------------------------------------
; N1: memcpy in the inner loop -- real (unanalysable) memory write, must block.
; CHECK-LABEL: @memcpy_neg(
; CHECK-NOT: add nuw nsw i64 %i, 2
define void @memcpy_neg(ptr noalias %A, ptr noalias %B, i64 %N, i64 %M) {
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
  %row = mul i64 %i, %M
  %idx = add i64 %row, %j
  %b.ptr = getelementptr inbounds float, ptr %B, i64 %idx
  call void @llvm.memcpy.p0.p0.i64(ptr %b.ptr, ptr %a.idx, i64 4, i1 false)
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
; N2: memset in the inner loop -- real memory write, must block.
; CHECK-LABEL: @memset_neg(
; CHECK-NOT: add nuw nsw i64 %i, 2
define void @memset_neg(ptr noalias %A, ptr noalias %B, i64 %N, i64 %M) {
entry:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  br label %inner.preheader
inner.preheader:
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %inner.preheader ], [ %j.next, %inner.header ]
  %row = mul i64 %i, %M
  %idx = add i64 %row, %j
  %b.ptr = getelementptr inbounds float, ptr %B, i64 %idx
  call void @llvm.memset.p0.i64(ptr %b.ptr, i8 0, i64 4, i1 false)
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
; N3: volatile store -- not guaranteed to transfer to its successor (and not a
; simple load/store), must block.
; CHECK-LABEL: @volatile_neg(
; CHECK-NOT: add nuw nsw i64 %i, 2
define void @volatile_neg(ptr noalias %A, ptr noalias %B, i64 %N, i64 %M) {
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
  %row = mul i64 %i, %M
  %idx = add i64 %row, %j
  %b.ptr = getelementptr inbounds float, ptr %B, i64 %idx
  store volatile float %a.val, ptr %b.ptr, align 4
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
; N4: atomic store -- non-simple access, must block.
; CHECK-LABEL: @atomic_neg(
; CHECK-NOT: add nuw nsw i64 %i, 2
define void @atomic_neg(ptr noalias %A, ptr noalias %B, i64 %N, i64 %M) {
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
  %row = mul i64 %i, %M
  %idx = add i64 %row, %j
  %b.ptr = getelementptr inbounds float, ptr %B, i64 %idx
  store atomic float %a.val, ptr %b.ptr unordered, align 4
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
; N5: masked.store -- real (masked) memory write, must block.
; CHECK-LABEL: @masked_store_neg(
; CHECK-NOT: add nuw nsw i64 %i, 2
define void @masked_store_neg(ptr noalias %A, ptr noalias %B, i64 %N, i64 %M) {
entry:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  br label %inner.preheader
inner.preheader:
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %inner.preheader ], [ %j.next, %inner.header ]
  %a.idx = getelementptr inbounds <4 x float>, ptr %A, i64 %j
  %a.vec = load <4 x float>, ptr %a.idx, align 16
  %b.ptr = getelementptr inbounds <4 x float>, ptr %B, i64 %j
  call void @llvm.masked.store.v4f32.p0(<4 x float> %a.vec, ptr %b.ptr, i32 16, <4 x i1> <i1 true, i1 true, i1 true, i1 true>)
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
; N6: experimental.noalias.scope.decl -- only accesses inaccessible memory (so
; it passes the generic memory-effects test) BUT jamming would clone it without
; duplicating its !noalias scopes (UnrollAndJamLoop does not remap them),
; corrupting alias metadata. It is therefore explicitly excluded from the safe
; set and must keep blocking even with the flag.
; CHECK-LABEL: @noalias_scope_neg(
; CHECK-NOT: add nuw nsw i64 %i, 2
define void @noalias_scope_neg(ptr noalias %A, ptr noalias %B, i64 %N, i64 %M) {
entry:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  br label %inner.preheader
inner.preheader:
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %inner.preheader ], [ %j.next, %inner.header ]
  call void @llvm.experimental.noalias.scope.decl(metadata !0)
  %a.idx = getelementptr inbounds float, ptr %A, i64 %j
  %a.val = load float, ptr %a.idx, align 4
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
; N7: opaque external call -- unknown memory effects / may throw, must block.
; CHECK-LABEL: @unknown_call_neg(
; CHECK-NOT: add nuw nsw i64 %i, 2
define void @unknown_call_neg(ptr noalias %A, ptr noalias %B, i64 %N, i64 %M) {
entry:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  br label %inner.preheader
inner.preheader:
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %inner.preheader ], [ %j.next, %inner.header ]
  call void @ext_func()
  %a.idx = getelementptr inbounds float, ptr %A, i64 %j
  %a.val = load float, ptr %a.idx, align 4
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
; N8: a genuine UJ-illegal cross-iteration dependence alongside a safe assume.
; B[i+j] is a diagonal access: iteration (i, j) and iteration (i+1, j-1) touch
; the SAME element, i.e. the dependence has a negative inner direction, which
; jamming (which interleaves i and i+1 at equal j) would reorder. The assume is
; correctly ignored by the relaxed analysis, but the REAL dependence must still
; block -- proving we did not disable genuine dependence checking.
; CHECK-LABEL: @real_dep_with_assume_neg(
; CHECK-NOT: add nuw nsw i64 %i, 2
define void @real_dep_with_assume_neg(ptr noalias %A, ptr noalias %B, i64 %N, i64 %M) {
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
  %diag = add i64 %i, %j                                   ; NOTE: B[i+j] -> diagonal dep
  %b.ptr = getelementptr inbounds float, ptr %B, i64 %diag
  %b.old = load float, ptr %b.ptr, align 4
  %sum = fadd float %b.old, %a.val
  store float %sum, ptr %b.ptr, align 4
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

!0 = !{!1}
!1 = distinct !{!1, !2, !"scope"}
!2 = distinct !{!2, i1 false, !"domain"}
