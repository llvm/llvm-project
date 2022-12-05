; RUN: opt < %s -passes=loop-vectorize,dce,instcombine -force-vector-interleave=1 -force-vector-width=4 -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; The parallel loop has been invalidated by the new memory accesses introduced
; by reg2mem (Loop::isParallel() starts to return false). Ensure the loop is
; now non-vectorizable.

;CHECK-NOT: <4 x i32>
define void @parallel_loop(i32* nocapture %a, i32* nocapture %b) nounwind uwtable {
entry:
  %indvars.iv.next.reg2mem = alloca i64
  %indvars.iv.reg2mem = alloca i64
  %"reg2mem alloca point" = bitcast i32 0 to i32
  store i64 0, i64* %indvars.iv.reg2mem
  br label %for.body

for.body:                                         ; preds = %for.body.for.body_crit_edge, %entry
  %indvars.iv.reload = load i64, i64* %indvars.iv.reg2mem
  %arrayidx = getelementptr inbounds i32, i32* %b, i64 %indvars.iv.reload
  %0 = load i32, i32* %arrayidx, align 4, !llvm.access.group !4
  %arrayidx2 = getelementptr inbounds i32, i32* %a, i64 %indvars.iv.reload
  %1 = load i32, i32* %arrayidx2, align 4, !llvm.access.group !4
  %idxprom3 = sext i32 %1 to i64
  %arrayidx4 = getelementptr inbounds i32, i32* %a, i64 %idxprom3
  store i32 %0, i32* %arrayidx4, align 4, !llvm.access.group !4
  %indvars.iv.next = add i64 %indvars.iv.reload, 1
  ; A new store without the parallel metadata here:
  store i64 %indvars.iv.next, i64* %indvars.iv.next.reg2mem
  %indvars.iv.next.reload1 = load i64, i64* %indvars.iv.next.reg2mem
  %arrayidx6 = getelementptr inbounds i32, i32* %b, i64 %indvars.iv.next.reload1
  %2 = load i32, i32* %arrayidx6, align 4, !llvm.access.group !4
  store i32 %2, i32* %arrayidx2, align 4, !llvm.access.group !4
  %indvars.iv.next.reload = load i64, i64* %indvars.iv.next.reg2mem
  %lftr.wideiv = trunc i64 %indvars.iv.next.reload to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 512
  br i1 %exitcond, label %for.end, label %for.body.for.body_crit_edge, !llvm.loop !3

for.body.for.body_crit_edge:                      ; preds = %for.body
  %indvars.iv.next.reload2 = load i64, i64* %indvars.iv.next.reg2mem
  store i64 %indvars.iv.next.reload2, i64* %indvars.iv.reg2mem
  br label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

!3 = !{!3, !{!"llvm.loop.parallel_accesses", !4}}
!4 = distinct !{}
