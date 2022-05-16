; RUN: opt -S -loop-vectorize -instcombine -force-vector-interleave=1 -force-vector-width=4 -force-target-supports-scalable-vectors=true -scalable-vectorization=on < %s | FileCheck %s --check-prefix=CHECKUF1
; RUN: opt -S -loop-vectorize -instcombine -force-vector-interleave=2 -force-vector-width=4 -force-target-supports-scalable-vectors=true -scalable-vectorization=on < %s | FileCheck %s --check-prefix=CHECKUF2

; CHECKUF1: for.body.preheader:
; CHECKUF1-DAG: %wide.trip.count = zext i32 %N to i64
; CHECKUF1-DAG: %[[VSCALE:.*]] = call i64 @llvm.vscale.i64()
; CHECKUF1-DAG: %[[VSCALEX4:.*]] = shl i64 %[[VSCALE]], 2
; CHECKUF1-DAG: %min.iters.check = icmp ugt i64 %[[VSCALEX4]], %wide.trip.count

; CHECKUF1: vector.ph:
; CHECKUF1-DAG:  %[[VSCALE:.*]] = call i64 @llvm.vscale.i64()
; CHECKUF1-DAG:  %[[VSCALEX4:.*]] = shl i64 %[[VSCALE]], 2
; CHECKUF1-DAG:  %n.mod.vf = urem i64 %wide.trip.count, %[[VSCALEX4]]
; CHECKUF1:      %n.vec = sub nsw i64 %wide.trip.count, %n.mod.vf

; CHECKUF1: vector.body:
; CHECKUF1: %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; CHECKUF1: %[[IDXB:.*]] = getelementptr inbounds double, double* %b, i64 %index
; CHECKUF1: %[[IDXB_CAST:.*]] = bitcast double* %[[IDXB]] to <vscale x 4 x double>*
; CHECKUF1: %wide.load = load <vscale x 4 x double>, <vscale x 4 x double>* %[[IDXB_CAST]], align 8
; CHECKUF1: %[[FADD:.*]] = fadd <vscale x 4 x double> %wide.load, shufflevector (<vscale x 4 x double> insertelement (<vscale x 4 x double> poison, double 1.000000e+00, i32 0), <vscale x 4 x double> poison, <vscale x 4 x i32> zeroinitializer)
; CHECKUF1: %[[IDXA:.*]] = getelementptr inbounds double, double* %a, i64 %index
; CHECKUF1: %[[IDXA_CAST:.*]] = bitcast double* %[[IDXA]] to <vscale x 4 x double>*
; CHECKUF1: store <vscale x 4 x double> %[[FADD]], <vscale x 4 x double>* %[[IDXA_CAST]], align 8
; CHECKUF1: %[[VSCALE:.*]] = call i64 @llvm.vscale.i64()
; CHECKUF1: %[[VSCALEX4:.*]] = shl i64 %[[VSCALE]], 2
; CHECKUF1: %index.next = add nuw i64 %index, %[[VSCALEX4]]
; CHECKUF1: %[[CMP:.*]] = icmp eq i64 %index.next, %n.vec
; CHECKUF1: br i1 %[[CMP]], label %middle.block, label %vector.body, !llvm.loop !0


; For an interleave factor of 2, vscale is scaled by 8 instead of 4 (and thus shifted left by 3 instead of 2).
; There is also the increment for the next iteration, e.g. instead of indexing IDXB, it indexes at IDXB + vscale * 4.

; CHECKUF2: for.body.preheader:
; CHECKUF2-DAG: %wide.trip.count = zext i32 %N to i64
; CHECKUF2-DAG: %[[VSCALE:.*]] = call i64 @llvm.vscale.i64()
; CHECKUF2-DAG: %[[VSCALEX8:.*]] = shl i64 %[[VSCALE]], 3
; CHECKUF2-DAG: %min.iters.check = icmp ugt i64 %[[VSCALEX8]], %wide.trip.count

; CHECKUF2: vector.ph:
; CHECKUF2-DAG:  %[[VSCALE:.*]] = call i64 @llvm.vscale.i64()
; CHECKUF2-DAG:  %[[VSCALEX8:.*]] = shl i64 %[[VSCALE]], 3
; CHECKUF2-DAG:  %n.mod.vf = urem i64 %wide.trip.count, %[[VSCALEX8]]
; CHECKUF2:      %n.vec = sub nsw i64 %wide.trip.count, %n.mod.vf

; CHECKUF2: vector.body:
; CHECKUF2: %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; CHECKUF2: %[[IDXB:.*]] = getelementptr inbounds double, double* %b, i64 %index
; CHECKUF2: %[[IDXB_CAST:.*]] = bitcast double* %[[IDXB]] to <vscale x 4 x double>*
; CHECKUF2: %wide.load = load <vscale x 4 x double>, <vscale x 4 x double>* %[[IDXB_CAST]], align 8
; CHECKUF2: %[[VSCALE:.*]] = call i32 @llvm.vscale.i32()
; CHECKUF2: %[[VSCALE2:.*]] = shl i32 %[[VSCALE]], 2
; CHECKUF2: %[[VSCALE2_EXT:.*]] = sext i32 %[[VSCALE2]] to i64
; CHECKUF2: %[[IDXB_NEXT:.*]] = getelementptr inbounds double, double* %[[IDXB]], i64 %[[VSCALE2_EXT]]
; CHECKUF2: %[[IDXB_NEXT_CAST:.*]] = bitcast double* %[[IDXB_NEXT]] to <vscale x 4 x double>*
; CHECKUF2: %wide.load{{[0-9]+}} = load <vscale x 4 x double>, <vscale x 4 x double>* %[[IDXB_NEXT_CAST]], align 8
; CHECKUF2: %[[FADD:.*]] = fadd <vscale x 4 x double> %wide.load, shufflevector (<vscale x 4 x double> insertelement (<vscale x 4 x double> poison, double 1.000000e+00, i32 0), <vscale x 4 x double> poison, <vscale x 4 x i32> zeroinitializer)
; CHECKUF2: %[[FADD_NEXT:.*]] = fadd <vscale x 4 x double> %wide.load{{[0-9]+}}, shufflevector (<vscale x 4 x double> insertelement (<vscale x 4 x double> poison, double 1.000000e+00, i32 0), <vscale x 4 x double> poison, <vscale x 4 x i32> zeroinitializer)
; CHECKUF2: %[[IDXA:.*]] = getelementptr inbounds double, double* %a, i64 %index
; CHECKUF2: %[[IDXA_CAST:.*]] = bitcast double* %[[IDXA]] to <vscale x 4 x double>*
; CHECKUF2: store <vscale x 4 x double> %[[FADD]], <vscale x 4 x double>* %[[IDXA_CAST]], align 8
; CHECKUF2: %[[VSCALE:.*]] = call i32 @llvm.vscale.i32()
; CHECKUF2: %[[VSCALE2:.*]] = shl i32 %[[VSCALE]], 2
; CHECKUF2: %[[VSCALE2_EXT:.*]] = sext i32 %[[VSCALE2]] to i64
; CHECKUF2: %[[IDXA_NEXT:.*]] = getelementptr inbounds double, double* %[[IDXA]], i64 %[[VSCALE2_EXT]]
; CHECKUF2: %[[IDXA_NEXT_CAST:.*]] = bitcast double* %[[IDXA_NEXT]] to <vscale x 4 x double>*
; CHECKUF2: store <vscale x 4 x double> %[[FADD_NEXT]], <vscale x 4 x double>* %[[IDXA_NEXT_CAST]], align 8
; CHECKUF2: %[[VSCALE:.*]] = call i64 @llvm.vscale.i64()
; CHECKUF2: %[[VSCALEX8:.*]] = shl i64 %[[VSCALE]], 3
; CHECKUF2: %index.next = add nuw i64 %index, %[[VSCALEX8]]
; CHECKUF2: %[[CMP:.*]] = icmp eq i64 %index.next, %n.vec
; CHECKUF2: br i1 %[[CMP]], label %middle.block, label %vector.body, !llvm.loop !0

define void @loop(i32 %N, double* nocapture %a, double* nocapture readonly %b) {
entry:
  %cmp7 = icmp sgt i32 %N, 0
  br i1 %cmp7, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %N to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, double* %b, i64 %indvars.iv
  %0 = load double, double* %arrayidx, align 8
  %add = fadd double %0, 1.000000e+00
  %arrayidx2 = getelementptr inbounds double, double* %a, i64 %indvars.iv
  store double %add, double* %arrayidx2, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !llvm.loop !1
}

!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
