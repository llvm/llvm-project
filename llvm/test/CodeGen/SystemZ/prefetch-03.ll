; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 -prefetch-distance=50 \
; RUN:   -loop-prefetch-writes -stop-after=loop-data-prefetch | FileCheck %s
;
; Check that prefetches are emitted in a position that is executed each
; iteration for each targeted memory instruction. The two stores in %true and
; %false are within one cache line in memory, so they should get a single
; prefetch in %for.body.
;
; CHECK-LABEL: for.body
; CHECK: call void @llvm.prefetch.p0i8(i8* {{.*}}, i32 0
; CHECK: call void @llvm.prefetch.p0i8(i8* {{.*}}, i32 1
; CHECK-LABEL: true
; CHECK-LABEL: false
; CHECK-LABEL: latch

define void @fun(i32* nocapture %Src, i32* nocapture readonly %Dst) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next.9, %latch ]
  %arrayidx = getelementptr inbounds i32, i32* %Dst, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %cmp = icmp sgt i32 %0, 0
  br i1 %cmp, label %true, label %false

true:  
  %arrayidx2 = getelementptr inbounds i32, i32* %Src, i64 %indvars.iv
  store i32 %0, i32* %arrayidx2, align 4
  br label %latch

false:
  %a = add i64 %indvars.iv, 8
  %arrayidx3 = getelementptr inbounds i32, i32* %Src, i64 %a
  store i32 %0, i32* %arrayidx3, align 4
  br label %latch

latch:
  %indvars.iv.next.9 = add nuw nsw i64 %indvars.iv, 1600
  %cmp.9 = icmp ult i64 %indvars.iv.next.9, 11200
  br i1 %cmp.9, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void
}

