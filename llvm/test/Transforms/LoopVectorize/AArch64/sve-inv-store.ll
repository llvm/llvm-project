; RUN: opt -loop-vectorize -scalable-vectorization=on -S < %s | FileCheck %s

target triple = "aarch64-unknown-linux-gnu"

define void @inv_store_i16(i16* noalias %dst, i16* noalias readonly %src, i64 %N) #0 {
; CHECK-LABEL: @inv_store_i16(
; CHECK:       vector.ph:
; CHECK:         %[[TMP1:.*]] = insertelement <vscale x 4 x i16*> poison, i16* %dst, i32 0
; CHECK-NEXT:    %[[SPLAT_PTRS:.*]] = shufflevector <vscale x 4 x i16*> %[[TMP1]], <vscale x 4 x i16*> poison, <vscale x 4 x i32> zeroinitializer
; CHECK:       vector.body:
; CHECK:         %[[VECLOAD:.*]] = load <vscale x 4 x i16>, <vscale x 4 x i16>* %{{.*}}, align 2
; CHECK-NEXT:    call void @llvm.masked.scatter.nxv4i16.nxv4p0i16(<vscale x 4 x i16> %[[VECLOAD]], <vscale x 4 x i16*> %[[SPLAT_PTRS]], i32 2
entry:
  br label %for.body14

for.body14:                                       ; preds = %for.body14, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body14 ]
  %arrayidx = getelementptr inbounds i16, i16* %src, i64 %indvars.iv
  %ld = load i16, i16* %arrayidx
  store i16 %ld, i16* %dst, align 2
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %N
  br i1 %exitcond.not, label %for.inc24, label %for.body14, !llvm.loop !0

for.inc24:                                        ; preds = %for.body14, %for.body
  ret void
}


define void @cond_inv_store_i32(i32* noalias %dst, i32* noalias readonly %src, i64 %N) #0 {
; CHECK-LABEL: @cond_inv_store_i32(
; CHECK:       vector.ph:
; CHECK:         %[[TMP1:.*]] = insertelement <vscale x 4 x i32*> poison, i32* %dst, i32 0
; CHECK-NEXT:    %[[SPLAT_PTRS:.*]] = shufflevector <vscale x 4 x i32*> %[[TMP1]], <vscale x 4 x i32*> poison, <vscale x 4 x i32> zeroinitializer
; CHECK:       vector.body:
; CHECK:         %[[VECLOAD:.*]] = load <vscale x 4 x i32>, <vscale x 4 x i32>* %{{.*}}, align 4
; CHECK-NEXT:    %[[MASK:.*]] = icmp sgt <vscale x 4 x i32> %[[VECLOAD]], shufflevector (<vscale x 4 x i32> insertelement (<vscale x 4 x i32> poison, i32 0, i32 0), <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-NEXT:    call void @llvm.masked.scatter.nxv4i32.nxv4p0i32(<vscale x 4 x i32> %[[VECLOAD]], <vscale x 4 x i32*> %[[SPLAT_PTRS]], i32 4, <vscale x 4 x i1> %[[MASK]])
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.inc
  %i.09 = phi i64 [ %inc, %for.inc ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %src, i64 %i.09
  %0 = load i32, i32* %arrayidx, align 4
  %cmp1 = icmp sgt i32 %0, 0
  br i1 %cmp1, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  store i32 %0, i32* %dst, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %inc = add nuw nsw i64 %i.09, 1
  %exitcond.not = icmp eq i64 %inc, %N
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:                                          ; preds = %for.inc, %entry
  ret void
}

attributes #0 = { "target-features"="+neon,+sve" vscale_range(0, 16) }

!0 = distinct !{!0, !1, !2, !3, !4, !5}
!1 = !{!"llvm.loop.mustprogress"}
!2 = !{!"llvm.loop.vectorize.width", i32 4}
!3 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
!4 = !{!"llvm.loop.vectorize.enable", i1 true}
!5 = !{!"llvm.loop.interleave.count", i32 1}

