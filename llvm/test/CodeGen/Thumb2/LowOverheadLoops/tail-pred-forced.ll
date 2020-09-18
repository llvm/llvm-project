; RUN: opt -mtriple=thumbv8.1m.main -mve-tail-predication -tail-predication=enabled -mattr=+mve,+lob %s -S -o - | FileCheck %s --check-prefixes=CHECK,ENABLED
; RUN: opt -mtriple=thumbv8.1m.main -mve-tail-predication -tail-predication=force-enabled -mattr=+mve,+lob %s -S -o - | FileCheck %s --check-prefixes=CHECK,FORCED

; CHECK-LABEL: set_iterations_not_rounded_up
;
; ENABLED:     call <4 x i1> @llvm.get.active.lane.mask
; ENABLED-NOT: vctp
;
; FORCED-NOT:  call <4 x i1> @llvm.get.active.lane.mask
; FORCED:      vctp
;
; CHECK:       ret void
;
define dso_local void @set_iterations_not_rounded_up(i32* noalias nocapture %A, i32* noalias nocapture readonly %B, i32* noalias nocapture readonly %C, i32 %N) local_unnamed_addr #0 {
entry:
  %cmp8 = icmp sgt i32 %N, 0

; Here, v5 which is used in set.loop.iterations which is usually rounded up to
; a next multiple of the VF when emitted from the vectoriser, which means a
; bound can be put on this expression. Without this, we can't, and should flag
; this as potentially overflow behaviour.

  %v5 = add nuw nsw i32 %N, 1
  br i1 %cmp8, label %vector.ph, label %for.cond.cleanup

vector.ph:                                        ; preds = %entry
  %trip.count.minus.1 = add i32 %N, -1
  call void @llvm.set.loop.iterations.i32(i32 %v5)
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %lsr.iv17 = phi i32* [ %scevgep18, %vector.body ], [ %A, %vector.ph ]
  %lsr.iv14 = phi i32* [ %scevgep15, %vector.body ], [ %C, %vector.ph ]
  %lsr.iv = phi i32* [ %scevgep, %vector.body ], [ %B, %vector.ph ]
  %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %v6 = phi i32 [ %v5, %vector.ph ], [ %v8, %vector.body ]
  %lsr.iv13 = bitcast i32* %lsr.iv to <4 x i32>*
  %lsr.iv1416 = bitcast i32* %lsr.iv14 to <4 x i32>*
  %lsr.iv1719 = bitcast i32* %lsr.iv17 to <4 x i32>*
  %active.lane.mask = call <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32 %index, i32 %N)
  %wide.masked.load = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %lsr.iv13, i32 4, <4 x i1> %active.lane.mask, <4 x i32> undef)
  %wide.masked.load12 = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %lsr.iv1416, i32 4, <4 x i1> %active.lane.mask, <4 x i32> undef)
  %v7 = add nsw <4 x i32> %wide.masked.load12, %wide.masked.load
  call void @llvm.masked.store.v4i32.p0v4i32(<4 x i32> %v7, <4 x i32>* %lsr.iv1719, i32 4, <4 x i1> %active.lane.mask)
  %index.next = add i32 %index, 4
  %scevgep = getelementptr i32, i32* %lsr.iv, i32 4
  %scevgep15 = getelementptr i32, i32* %lsr.iv14, i32 4
  %scevgep18 = getelementptr i32, i32* %lsr.iv17, i32 4
  %v8 = call i32 @llvm.loop.decrement.reg.i32(i32 %v6, i32 1)
  %v9 = icmp ne i32 %v8, 0
  br i1 %v9, label %vector.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %vector.body, %entry
  ret void
}

declare <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>*, i32 immarg, <4 x i1>, <4 x i32>)
declare void @llvm.masked.store.v4i32.p0v4i32(<4 x i32>, <4 x i32>*, i32 immarg, <4 x i1>)
declare void @llvm.set.loop.iterations.i32(i32)
declare i32 @llvm.loop.decrement.reg.i32(i32, i32)
declare <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32, i32)
