; RUN: opt -passes=loop-vectorize -force-vector-interleave=1 -vectorize-vector-loops -mtriple=aarch64 -mattr=+sve2p1 -S < %s | FileCheck %s

; Make sure phi nodes that are operands of shufflevectors get correctly blended.

define void @blend_simple_phi(ptr noalias %dst, i64 %conv4.i11, <8 x i16> %in) {
; CHECK-LABEL: define void @blend_simple_phi(
; CHECK:     vector.ph:
; CHECK:       [[BROADCAST:%.*]] = call <vscale x 8 x i16> @llvm.vector.broadcast.nxv8i16.v8i16(<8 x i16> %in)
; CHECK:       segmented.shuffle.nxv8i16.v8i32(<vscale x 8 x i16> [[BROADCAST]]
entry:
  br label %do.body29.i

do.body29.i:                                      ; preds = %if.end57.i, %entry
  %indvars.iv78.i = phi i64 [ 0, %entry ], [ %indvars.iv.next79.i, %if.end57.i ]
  br label %if.end57.i

if.end57.i:                                       ; preds = %do.body29.i
  %a030.1.i = phi <8 x i16> [ %in, %do.body29.i ]
  %shuffle.i4.i.i44 = shufflevector <8 x i16> %a030.1.i, <8 x i16> %a030.1.i, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 0, i32 2, i32 4, i32 6>
  %arrayidx2 = getelementptr inbounds <8 x i16>, ptr %dst, i64 %indvars.iv78.i
  store <8 x i16> %shuffle.i4.i.i44, ptr %arrayidx2, align 2
  %indvars.iv.next79.i = add i64 %indvars.iv78.i, 1
  %cmp87.i = icmp slt i64 %indvars.iv78.i, %conv4.i11
  br i1 %cmp87.i, label %do.body29.i, label %loopexit.i

loopexit.i:                              ; preds = %if.end57.i
  ret void
}
