; RUN: opt -passes=loop-vectorize -vectorize-vector-loops -lv-strided-pointer-ivs \
; RUN:     -force-vector-interleave=1 -scalable-vectorization=on -force-vector-width=1 \
; RUN:     -mtriple=aarch64 -mattr=+sve2p1 -S < %s | FileCheck %s --check-prefixes=CHECK,VF1
; RUN: opt -passes=loop-vectorize -vectorize-vector-loops -lv-strided-pointer-ivs \
; RUN:     -force-vector-interleave=1 -scalable-vectorization=on -force-vector-width=2 \
; RUN:     -mtriple=aarch64 -mattr=+sve2p1 -S < %s | FileCheck %s --check-prefixes=CHECK,VF2

; Check that gathers cannot be re-vectorised for VF = vscale x 2.
; This is a hard limitation of .segment intrinsics which expect types like <vscale x SegmentTy>.

define void @gather_128(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b, i64 %stride) {
; CHECK-LABEL: define void @gather_128(
; CHECK: call <vscale x 8 x i16> @llvm.masked.segment.gather
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %offset = mul i64 %indvars.iv, %stride
  %arrayidx0 = getelementptr inbounds i16, ptr %b, i64 %offset
  %0 = load <8 x i16>, ptr %arrayidx0, align 16
  %result = add <8 x i16> %0, splat (i16 1)
  %arrayidx2 = getelementptr inbounds <8 x i16>, ptr %a, i64 %indvars.iv
  store <8 x i16> %result, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @gather_64(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b, i64 %stride) {
; CHECK-LABEL: define void @gather_64(
; VF1: call <vscale x 4 x i16> @llvm.masked.segment.gather
; VF2-NOT: vector.body
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %offset = mul i64 %indvars.iv, %stride
  %arrayidx0 = getelementptr inbounds i16, ptr %b, i64 %offset
  %0 = load <4 x i16>, ptr %arrayidx0, align 8
  %result = add <4 x i16> %0, splat (i16 1)
  %arrayidx2 = getelementptr inbounds <4 x i16>, ptr %a, i64 %indvars.iv
  store <4 x i16> %result, ptr %arrayidx2, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
