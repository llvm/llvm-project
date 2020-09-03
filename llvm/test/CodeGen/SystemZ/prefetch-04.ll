; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 -prefetch-distance=20 \
; RUN:   -loop-prefetch-writes -stop-after=loop-data-prefetch | FileCheck %s
;
; Check that for a load followed by a store to the same address gets a single
; write prefetch.
;
; CHECK-LABEL: for.body
; CHECK: call void @llvm.prefetch.p0i8(i8* %scevgep{{.*}}, i32 1, i32 3, i32 1
; CHECK-not: call void @llvm.prefetch

define void @fun(i32* nocapture %Src, i32* nocapture readonly %Dst) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next.9, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %Dst, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %a = add i32 %0, 128
  store i32 %a, i32* %arrayidx, align 4
  %indvars.iv.next.9 = add nuw nsw i64 %indvars.iv, 1600
  %cmp.9 = icmp ult i64 %indvars.iv.next.9, 11200
  br i1 %cmp.9, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void
}

