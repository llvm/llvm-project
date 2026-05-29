; RUN: opt < %s -passes=loop-vectorize -enable-masked-invariant-load -force-vector-width=4 \
; RUN:     -force-vector-interleave=1 -mtriple=x86_64-unknown-linux-gnu -mattr=avx2 -S \
; RUN:     | FileCheck %s

;; Negative test for CM_CondInvar.
;;
;; The conditional load's pointer (%src + i) is loop-VARIANT, so even with
;; -enable-masked-invariant-load enabled, getConditionalInvarCost() returns
;; Invalid and the cost model picks the regular consecutive masked load
;; (CM_Widen).
;;
;;   for (i = 0; i < n; ++i)
;;     if (cond[i])
;;       a[i] = src[i];
;;
;; The output:
;;   * Must contain a regular widened consecutive masked load on src + index.
;;   * Must NOT contain any AnyOf-style i1 OR-reduction.
;;   * Must NOT contain a broadcast.lane shuffle (the CondInvar lowering
;;     marker).
;;   * Must NOT contain a masked.load whose pointer is the loop-invariant
;;     %src argument.

define void @cond_load_not_invariant(ptr noalias %a, ptr noalias %src, ptr noalias %cond, i64 %n) {
; CHECK-LABEL: define void @cond_load_not_invariant(

; The vector body widens the consecutive masked load on src[index].
; CHECK:       vector.body:
; CHECK:         [[SRC_GEP_V:%.*]] = getelementptr i32, ptr %src, i64 {{%.*}}
; CHECK:         call <4 x i32> @llvm.masked.load.v4i32.p0(ptr align 4 [[SRC_GEP_V]],

; The CondInvar lowering must NOT be applied.
; CHECK-NOT:     call i1 @llvm.vector.reduce.or
; CHECK-NOT:     %broadcast.lane = shufflevector
; CHECK-NOT:     call <4 x i32> @llvm.masked.load.v4i32.p0(ptr align 4 %src,

; CHECK:       middle.block:
; CHECK:         ret void

entry:
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.inc ]
  %cond.gep = getelementptr inbounds i32, ptr %cond, i64 %i
  %cv = load i32, ptr %cond.gep, align 4
  %tobool = icmp ne i32 %cv, 0
  br i1 %tobool, label %if.then, label %for.inc

if.then:
  %src.gep = getelementptr inbounds i32, ptr %src, i64 %i
  %v = load i32, ptr %src.gep, align 4
  %a.gep = getelementptr inbounds i32, ptr %a, i64 %i
  store i32 %v, ptr %a.gep, align 4
  br label %for.inc

for.inc:
  %i.next = add nuw nsw i64 %i, 1
  %exitcond = icmp eq i64 %i.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}
