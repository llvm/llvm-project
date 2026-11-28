; RUN: opt %loadNPMPolly -plugin-arg=Polly,-polly-parallel -plugin-arg=Polly,-polly-parallel-force '-passes=polly' -S -verify-dom-info < %s | FileCheck %s
; RUN: opt %loadNPMPolly -plugin-arg=Polly,-polly-parallel -plugin-arg=Polly,-polly-parallel-force '-passes=polly' -plugin-arg=Polly,-polly-omp-backend=LLVM -S -verify-dom-info < %s | FileCheck %s

; https://github.com/llvm/llvm-project/issues/234413
;
; The scenario requires Polly's default optimization pipeline (-passes=polly);
; with polly<no-default-opts> the loop is not parallelized and the bug does
; not trigger.
;
; %val is defined outside the SCoP and used only as an incoming value of %phi
; in %inner.latch, the exit block of the non-affine subregion
; %inner.header..%inner.latch. Polly models %phi as a PHI write access on the
; subregion statement with %val as one of its incoming values, but no
; instruction of any statement in the outlined subtree references %val.
; Consequently, the reference collection for parallel code generation never
; passed %val to the outlined subfunction, and RegionGenerator::buildExitPHI()
; hit the ScopExpander::isInOrigRegion assertion (or, with NDEBUG, generated
; invalid cross-function IR).
;
; The value must now be passed via polly.par.userContext and reloaded inside
; the subfunction.

; CHECK: store ptr %val, ptr %polly.subfn.storeaddr.val
; CHECK-LABEL: define internal void @foo_polly_subfn
; CHECK: %polly.subfunc.arg.val = load ptr, ptr
; CHECK: %polly.phi = phi ptr [ %loaded_p_scalar_, {{.*}} ], [ %polly.subfunc.arg.val, {{.*}} ]

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

define void @foo(ptr %src, ptr %dst, i1 %cond) {
entry:
  %val = load ptr, ptr %src, align 8
  br label %outer.header

outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %offset = shl nuw nsw i64 %i, 7
  %row = getelementptr ptr, ptr %dst, i64 %offset
  br label %inner.header

outer.exit:
  ret void

outer.latch:
  %i.next = add nuw nsw i64 %i, 1
  %outer.cmp = icmp eq i64 %i, 100
  br i1 %outer.cmp, label %outer.exit, label %outer.header

inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.latch ]
  br i1 %cond, label %inner.then, label %inner.latch

inner.then:
  %loaded = load ptr, ptr %src, align 8
  br label %inner.latch

inner.latch:
  %phi = phi ptr [ %loaded, %inner.then ], [ %val, %inner.header ]
  %gep = getelementptr ptr, ptr %row, i64 %j
  store ptr %phi, ptr %gep, align 8
  %j.next = add nuw nsw i64 %j, 1
  %inner.cmp = icmp eq i64 %j, 100
  br i1 %inner.cmp, label %outer.latch, label %inner.header
}
