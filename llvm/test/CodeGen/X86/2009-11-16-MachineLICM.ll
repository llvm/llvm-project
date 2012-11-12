; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s
; rdar://7395200

@g = common global [4 x float] zeroinitializer, align 16 ; <[4 x float]*> [#uses=4]

define void @foo(i32 %n, float* nocapture %x) nounwind ssp {
entry:
; CHECK: foo:
  %0 = icmp sgt i32 %n, 0                         ; <i1> [#uses=1]
  br i1 %0, label %bb.nph, label %return

bb.nph:                                           ; preds = %entry
; CHECK: movq _g@GOTPCREL(%rip), [[REG:%[a-z]+]]
  %tmp = zext i32 %n to i64                       ; <i64> [#uses=1]
  br label %bb

bb:                                               ; preds = %bb, %bb.nph
; CHECK: LBB0_2:
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %bb ] ; <i64> [#uses=2]
  %tmp9 = shl i64 %indvar, 2                      ; <i64> [#uses=4]
  %tmp1016 = or i64 %tmp9, 1                      ; <i64> [#uses=1]
  %scevgep = getelementptr float* %x, i64 %tmp1016 ; <float*> [#uses=1]
  %tmp1117 = or i64 %tmp9, 2                      ; <i64> [#uses=1]
  %scevgep12 = getelementptr float* %x, i64 %tmp1117 ; <float*> [#uses=1]
  %tmp1318 = or i64 %tmp9, 3                      ; <i64> [#uses=1]
  %scevgep14 = getelementptr float* %x, i64 %tmp1318 ; <float*> [#uses=1]
  %x_addr.03 = getelementptr float* %x, i64 %tmp9 ; <float*> [#uses=1]
  %1 = load float* getelementptr inbounds ([4 x float]* @g, i64 0, i64 0), align 16 ; <float> [#uses=1]
  store float %1, float* %x_addr.03, align 4
  %2 = load float* getelementptr inbounds ([4 x float]* @g, i64 0, i64 1), align 4 ; <float> [#uses=1]
  store float %2, float* %scevgep, align 4
  %3 = load float* getelementptr inbounds ([4 x float]* @g, i64 0, i64 2), align 8 ; <float> [#uses=1]
  store float %3, float* %scevgep12, align 4
  %4 = load float* getelementptr inbounds ([4 x float]* @g, i64 0, i64 3), align 4 ; <float> [#uses=1]
  store float %4, float* %scevgep14, align 4
  %indvar.next = add i64 %indvar, 1               ; <i64> [#uses=2]
  %exitcond = icmp eq i64 %indvar.next, %tmp      ; <i1> [#uses=1]
  br i1 %exitcond, label %return, label %bb

return:                                           ; preds = %bb, %entry
  ret void
}
