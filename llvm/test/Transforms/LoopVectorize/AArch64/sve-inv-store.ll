; RUN: opt -loop-vectorize -scalable-vectorization=on -S < %s | FileCheck %s

target triple = "aarch64-unknown-linux-gnu"

define void @inv_store_i16(i16* noalias %dst, i16* noalias readonly %src, i64 %N) #0 {
; CHECK-LABEL: @inv_store_i16(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP1:%.*]] = mul i64 [[TMP0]], 4
; CHECK-NEXT:    [[MIN_ITERS_CHECK:%.*]] = icmp ult i64 [[N:%.*]], [[TMP1]]
; CHECK-NEXT:    br i1 [[MIN_ITERS_CHECK]], label [[SCALAR_PH:%.*]], label [[VECTOR_PH:%.*]]
; CHECK:       vector.ph:
; CHECK-NEXT:    [[TMP2:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP3:%.*]] = mul i64 [[TMP2]], 4
; CHECK-NEXT:    [[N_MOD_VF:%.*]] = urem i64 [[N]], [[TMP3]]
; CHECK-NEXT:    [[N_VEC:%.*]] = sub i64 [[N]], [[N_MOD_VF]]
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP4:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP5:%.*]] = getelementptr inbounds i16, i16* [[SRC:%.*]], i64 [[TMP4]]
; CHECK-NEXT:    [[TMP6:%.*]] = getelementptr inbounds i16, i16* [[TMP5]], i32 0
; CHECK-NEXT:    [[TMP7:%.*]] = bitcast i16* [[TMP6]] to <vscale x 4 x i16>*
; CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <vscale x 4 x i16>, <vscale x 4 x i16>* [[TMP7]], align 2
; CHECK-NEXT:    [[TMP8:%.*]] = call i32 @llvm.vscale.i32()
; CHECK-NEXT:    [[TMP9:%.*]] = mul i32 [[TMP8]], 4
; CHECK-NEXT:    [[TMP10:%.*]] = sub i32 [[TMP9]], 1
; CHECK-NEXT:    [[TMP11:%.*]] = extractelement <vscale x 4 x i16> [[WIDE_LOAD]], i32 [[TMP10]]
; CHECK-NEXT:    store i16 [[TMP11]], i16* [[DST:%.*]], align 2
; CHECK-NEXT:    [[TMP12:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP13:%.*]] = mul i64 [[TMP12]], 4
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], [[TMP13]]
; CHECK-NEXT:    [[TMP14:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[TMP14]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP0:![0-9]+]]
;
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

define void @uniform_store_i1(i1* noalias %dst, i64* noalias %start, i64 %N) #0 {
; CHECK-LABEL: @uniform_store_i1(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[N:%.*]], 1
; CHECK-NEXT:    [[TMP1:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP2:%.*]] = mul i64 [[TMP1]], 2
; CHECK-NEXT:    [[MIN_ITERS_CHECK:%.*]] = icmp ult i64 [[TMP0]], [[TMP2]]
; CHECK-NEXT:    br i1 [[MIN_ITERS_CHECK]], label [[SCALAR_PH:%.*]], label [[VECTOR_PH:%.*]]
; CHECK:       vector.ph:
; CHECK-NEXT:    [[TMP3:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP4:%.*]] = mul i64 [[TMP3]], 2
; CHECK-NEXT:    [[N_MOD_VF:%.*]] = urem i64 [[TMP0]], [[TMP4]]
; CHECK-NEXT:    [[N_VEC:%.*]] = sub i64 [[TMP0]], [[N_MOD_VF]]
; CHECK-NEXT:    [[IND_END:%.*]] = getelementptr i64, i64* [[START:%.*]], i64 [[N_VEC]]
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <vscale x 2 x i64*> poison, i64* [[START]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <vscale x 2 x i64*> [[BROADCAST_SPLATINSERT]], <vscale x 2 x i64*> poison, <vscale x 2 x i32> zeroinitializer
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP5:%.*]] = call <vscale x 2 x i64> @llvm.experimental.stepvector.nxv2i64()
; CHECK-NEXT:    [[DOTSPLATINSERT:%.*]] = insertelement <vscale x 2 x i64> poison, i64 [[INDEX]], i32 0
; CHECK-NEXT:    [[DOTSPLAT:%.*]] = shufflevector <vscale x 2 x i64> [[DOTSPLATINSERT]], <vscale x 2 x i64> poison, <vscale x 2 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP6:%.*]] = add <vscale x 2 x i64> shufflevector (<vscale x 2 x i64> insertelement (<vscale x 2 x i64> poison, i64 0, i32 0), <vscale x 2 x i64> poison, <vscale x 2 x i32> zeroinitializer), [[TMP5]]
; CHECK-NEXT:    [[TMP7:%.*]] = add <vscale x 2 x i64> [[DOTSPLAT]], [[TMP6]]
; CHECK-NEXT:    [[NEXT_GEP:%.*]] = getelementptr i64, i64* [[START]], <vscale x 2 x i64> [[TMP7]]
; CHECK-NEXT:    [[TMP8:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[NEXT_GEP2:%.*]] = getelementptr i64, i64* [[START]], i64 [[TMP8]]
; CHECK-NEXT:    [[TMP9:%.*]] = add i64 [[INDEX]], 1
; CHECK-NEXT:    [[NEXT_GEP3:%.*]] = getelementptr i64, i64* [[START]], i64 [[TMP9]]
; CHECK-NEXT:    [[TMP10:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP11:%.*]] = getelementptr i64, i64* [[NEXT_GEP2]], i32 0
; CHECK-NEXT:    [[TMP12:%.*]] = bitcast i64* [[TMP11]] to <vscale x 2 x i64>*
; CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <vscale x 2 x i64>, <vscale x 2 x i64>* [[TMP12]], align 4
; CHECK-NEXT:    [[TMP13:%.*]] = getelementptr inbounds i64, <vscale x 2 x i64*> [[NEXT_GEP]], i64 1
; CHECK-NEXT:    [[TMP14:%.*]] = icmp eq <vscale x 2 x i64*> [[TMP13]], [[BROADCAST_SPLAT]]
; CHECK-NEXT:    [[TMP15:%.*]] = call i32 @llvm.vscale.i32()
; CHECK-NEXT:    [[TMP16:%.*]] = mul i32 [[TMP15]], 2
; CHECK-NEXT:    [[TMP17:%.*]] = sub i32 [[TMP16]], 1
; CHECK-NEXT:    [[TMP18:%.*]] = extractelement <vscale x 2 x i1> [[TMP14]], i32 [[TMP17]]
; CHECK-NEXT:    store i1 [[TMP18]], i1* [[DST:%.*]], align 1
; CHECK-NEXT:    [[TMP19:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP20:%.*]] = mul i64 [[TMP19]], 2
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], [[TMP20]]
; CHECK-NEXT:    [[TMP21:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[TMP21]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP7:![0-9]+]]
;
entry:
  br label %for.body

for.body:
  %first.sroa = phi i64* [ %incdec.ptr, %for.body ], [ %start, %entry ]
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %iv.next = add i64 %iv, 1
  %0 = load i64, i64* %first.sroa
  %incdec.ptr = getelementptr inbounds i64, i64* %first.sroa, i64 1
  %cmp.not = icmp eq i64* %incdec.ptr, %start
  store i1 %cmp.not, i1* %dst
  %cmp = icmp ult i64 %iv, %N
  br i1 %cmp, label %for.body, label %end, !llvm.loop !6

end:
  ret void
}

; Ensure conditional i1 stores do not vectorize
define void @cond_store_i1(i1* noalias %dst, i8* noalias %start, i32 %cond, i64 %N) #0 {
; CHECK-LABEL: @cond_store_i1(
; CHECK-NOT:   vector.body
;
entry:
  br label %for.body

for.body:
  %first.sroa = phi i8* [ %incdec.ptr, %if.end ], [ null, %entry ]
  %incdec.ptr = getelementptr inbounds i8, i8* %first.sroa, i64 1
  %0 = load i8, i8* %incdec.ptr
  %tobool.not = icmp eq i8 %0, 10
  br i1 %tobool.not, label %if.end, label %if.then

if.then:
  %cmp.store = icmp eq i8* %start, %incdec.ptr
  store i1 %cmp.store, i1* %dst
  br label %if.end

if.end:
  %cmp.not = icmp eq i8* %incdec.ptr, %start
  br i1 %cmp.not, label %for.end, label %for.body

for.end:
  ret void
}

attributes #0 = { "target-features"="+neon,+sve" vscale_range(0, 16) }

!0 = distinct !{!0, !1, !2, !3, !4, !5}
!1 = !{!"llvm.loop.mustprogress"}
!2 = !{!"llvm.loop.vectorize.width", i32 4}
!3 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
!4 = !{!"llvm.loop.vectorize.enable", i1 true}
!5 = !{!"llvm.loop.interleave.count", i32 1}

!6 = distinct !{!6, !1, !7, !3, !4, !5}
!7 = !{!"llvm.loop.vectorize.width", i32 2}

