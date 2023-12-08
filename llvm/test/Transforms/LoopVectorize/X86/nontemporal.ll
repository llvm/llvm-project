; RUN: opt < %s -passes=loop-vectorize -force-vector-width=4 -S | FileCheck %s

; The three test-cases below are all based on modified versions of a simple copy-loop:
;
; void foo(unsigned *src, unsigned *dst, unsigned nElts) {
;   for (unsigned i = 0; i < nElts; ++i) {
;     unsigned tmp = src[i];
;     dst[i] = tmp;
;   }
; }
;
; In the first version, there are no nontemporal stores or loads, and so vectorization
; is safely done.
;
; In the second version, the store into dst[i] has the nontemporal hint.  The alignment
; on X86_64 for 'unsigned' is 4, so the vector store generally will not be aligned to the
; vector size (of 16 here).  Unaligned nontemporal vector stores are not supported on X86_64,
; and so the vectorization is suppressed (because when vectorizing it, the nontemoral hint
; would not be honored in the final code-gen).
;
; The third version is analogous to the second, except rather than the store, it is the
; load from 'src[i]' that has the nontemporal hint.  Vectorization is suppressed in this
; case because (like stores) unaligned nontemoral vector loads are not supported on X86_64.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64"

; CHECK-LABEL: @vectorTest(
define void @vectorTest(ptr noalias readonly %src, ptr noalias %dst, i32 %nElts) {
entry:
  %cmp8 = icmp eq i32 %nElts, 0
  br i1 %cmp8, label %for.cond.cleanup, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %nElts to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body, %for.body.preheader
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
; Check that we vectorized the load, and that there is no nontemporal hint.
; CHECK: %wide.load = load <4 x i32>, ptr %{{[0-9]+}}, align 4{{$}}
  %arrayidx = getelementptr inbounds i32, ptr %src, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
; Check that we vectorized the store, and that there is no nontemporal hint.
; CHECK: store <4 x i32> %wide.load, ptr %{{[0-9]+}}, align 4{{$}}
  %arrayidx2 = getelementptr inbounds i32, ptr %dst, i64 %indvars.iv
  store i32 %0, ptr %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; CHECK-LABEL: @vectorNTStoreTest(
; Check that the vectorized type of the store does not appear.
; CHECK-NOT: 4 x i32
define void @vectorNTStoreTest(ptr noalias readonly %src, ptr noalias %dst, i32 %nElts) {
entry:
  %cmp8 = icmp eq i32 %nElts, 0
  br i1 %cmp8, label %for.cond.cleanup, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %nElts to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body, %for.body.preheader
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %src, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %dst, i64 %indvars.iv
; Check that the store is not vectorized and that we don't lose the !nontemporal hint in it.
; CHECK: store i32 %{{[0-9]+}}, ptr %arrayidx2, align 4, !nontemporal !4
  store i32 %0, ptr %arrayidx2, align 4, !nontemporal !0
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; CHECK-LABEL: @vectorNTLoadTest(
; Check that the vectorized type of the load does not appear.
; CHECK-NOT: 4 x i32
define void @vectorNTLoadTest(ptr noalias readonly %src, ptr noalias %dst, i32 %nElts) {
entry:
  %cmp8 = icmp eq i32 %nElts, 0
  br i1 %cmp8, label %for.cond.cleanup, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %nElts to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body, %for.body.preheader
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %src, i64 %indvars.iv
; Check that the load is not vectorized and that we don't lose the !nontemporal hint in it.
; CHECK: load i32, ptr %arrayidx, align 4, !nontemporal !4
  %0 = load i32, ptr %arrayidx, align 4, !nontemporal !0
  %arrayidx2 = getelementptr inbounds i32, ptr %dst, i64 %indvars.iv
  store i32 %0, ptr %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

!0 = !{i32 1}
