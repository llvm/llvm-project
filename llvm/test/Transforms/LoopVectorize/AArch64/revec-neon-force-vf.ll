; RUN: opt -passes=loop-vectorize -force-vector-interleave=1 -vectorize-vector-loops \
; RUN:   -scalable-vectorization=on -force-vector-width=1 -mtriple=aarch64 -mattr=+sve2p1 \
; RUN:   -S < %s | FileCheck %s
; RUN: opt -passes=loop-vectorize -force-vector-interleave=1 -vectorize-vector-loops \
; RUN:   -scalable-vectorization=on -force-vector-width=2 -mtriple=aarch64 -mattr=+sve2p1 \
; RUN:   -S < %s | FileCheck %s

; Test that 128-bit NEON intrinsics cannot get re-vectorised with VF = vscale x 2

define void @uabd(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @uabd(
; CHECK: vector.body
; CHECK: load <vscale x 8 x i16>
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <8 x i16>, ptr %b, i64 %indvars.iv
  %0 = load <8 x i16>, ptr %arrayidx, align 16
  %result = call <8 x i16> @llvm.aarch64.neon.uabd.v8i16(<8 x i16> %0, <8 x i16> splat (i16 1))
  %arrayidx2 = getelementptr inbounds <8 x i16>, ptr %a, i64 %indvars.iv
  store <8 x i16> %result, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
