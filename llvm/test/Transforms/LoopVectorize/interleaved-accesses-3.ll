; RUN: opt -S -loop-vectorize -instcombine -force-vector-width=4 -force-vector-interleave=1 -enable-interleaved-mem-accesses=true < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Check that the interleaved-mem-access analysis currently does not create an 
; interleave group for access 'a' due to the possible pointer wrap-around.
;
; To begin with, in this test the candidate interleave group can be created 
; only when getPtrStride is called with Assume=true. Next, because
; the interleave-group of the loads is not full (has gaps), we also need to check 
; for possible pointer wrapping. Here we currently use Assume=false and as a 
; result cannot prove the transformation is safe and therefore invalidate the
; candidate interleave group.
;
; FIXME: This is a missed optimization. Once we use Assume=true here, we will
; not have to invalidate the group.

; void func(unsigned * __restrict a, unsigned * __restrict b, unsigned char x, unsigned char y) {
;  int i = 0;
;  for (unsigned char index = x; i < y; index +=2, ++i)
;    b[i] = a[index] * 2;
;
; }

; CHECK: vector.body:
; CHECK-NOT: %wide.vec = load <8 x i32>, <8 x i32>* {{.*}}, align 4
; CHECK-NOT: shufflevector <8 x i32> %wide.vec, <8 x i32> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>

define void @_Z4funcPjS_hh(i32* noalias nocapture readonly %a, i32* noalias nocapture %b, i8 zeroext %x, i8 zeroext %y) local_unnamed_addr {
entry:
  %cmp9 = icmp eq i8 %y, 0
  br i1 %cmp9, label %for.cond.cleanup, label %for.body.preheader

for.body.preheader:
  %wide.trip.count = zext i8 %y to i64
  br label %for.body

for.cond.cleanup.loopexit:
  br label %for.cond.cleanup

for.cond.cleanup:
  ret void

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %index.011 = phi i8 [ %add, %for.body ], [ %x, %for.body.preheader ]
  %idxprom = zext i8 %index.011 to i64
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %idxprom
  %0 = load i32, i32* %arrayidx, align 4
  %mul = shl i32 %0, 1
  %arrayidx2 = getelementptr inbounds i32, i32* %b, i64 %indvars.iv
  store i32 %mul, i32* %arrayidx2, align 4
  %add = add i8 %index.011, 2
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.cond.cleanup.loopexit, label %for.body
}
