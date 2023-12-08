; RUN: opt -S -passes=loop-vectorize,instcombine -force-vector-width=4 -force-vector-interleave=1 -enable-interleaved-mem-accesses=true < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Check that the interleaved-mem-access analysis currently does not create an
; interleave group for the access to array 'in' due to the possibly wrapping
; unsigned 'out_ix' index.
;
; In this test the interleave-group of the loads is not full (has gaps), so
; the wrapping checks are necessary. Here this cannot be done statically so
; runtime checks are needed, but with Assume=false getPtrStride cannot add
; runtime checks and as a result we can't create the interleave-group.
;
; FIXME: This is currently a missed optimization until we can use Assume=true
; with proper threshold checks. Once we do that the candidate interleave-group
; will not be invalidated by the wrapping checks.

; #include <stdlib.h>
; void test(ptr __restrict__ out, ptr __restrict__ in, size_t size)
; {
;    for (size_t out_offset = 0; out_offset < size; ++out_offset)
;      {
;        float t0 = in[2*out_offset];
;        out[out_offset] = t0;
;      }
; }

; CHECK: vector.body:
; CHECK-NOT: %wide.vec = load <8 x i32>, ptr {{.*}}, align 4
; CHECK-NOT: shufflevector <8 x i32> %wide.vec, <8 x i32> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>

define void @_Z4testPfS_m(ptr noalias nocapture %out, ptr noalias nocapture readonly %in, i64 %size) local_unnamed_addr {
entry:
  %cmp7 = icmp eq i64 %size, 0
  br i1 %cmp7, label %for.cond.cleanup, label %for.body.preheader

for.body.preheader:
  br label %for.body

for.cond.cleanup.loopexit:
  br label %for.cond.cleanup

for.cond.cleanup:
  ret void

for.body:
  %out_offset.08 = phi i64 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %mul = shl i64 %out_offset.08, 1
  %arrayidx = getelementptr inbounds float, ptr %in, i64 %mul
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds float, ptr %out, i64 %out_offset.08
  store i32 %0, ptr %arrayidx1, align 4
  %inc = add nuw i64 %out_offset.08, 1
  %exitcond = icmp eq i64 %inc, %size
  br i1 %exitcond, label %for.cond.cleanup.loopexit, label %for.body
}
