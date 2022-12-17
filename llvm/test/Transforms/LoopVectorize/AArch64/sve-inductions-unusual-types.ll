; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -S < %s -debug -prefer-predicate-over-epilogue=scalar-epilogue 2>%t | FileCheck %s
; RUN: cat %t | FileCheck %s --check-prefix=DEBUG

target triple = "aarch64-unknown-linux-gnu"

; DEBUG: Found an estimated cost of Invalid for VF vscale x 1 For instruction:   %indvars.iv1294 = phi i7 [ %indvars.iv.next1295, %for.body ], [ 0, %entry ]
; DEBUG: Found an estimated cost of Invalid for VF vscale x 1 For instruction:   %addi7 = add i7 %indvars.iv1294, 0
; DEBUG: Found an estimated cost of Invalid for VF vscale x 1 For instruction:   %indvars.iv.next1295 = add i7 %indvars.iv1294, 1

define void @induction_i7(i64* %dst) #0 {
; CHECK-LABEL: @induction_i7(
; CHECK:       vector.ph:
; CHECK:         [[TMP4:%.*]] = call <vscale x 2 x i8> @llvm.experimental.stepvector.nxv2i8()
; CHECK:         [[TMP5:%.*]] = trunc <vscale x 2 x i8> [[TMP4]] to <vscale x 2 x i7>
; CHECK-NEXT:    [[TMP6:%.*]] = add <vscale x 2 x i7> [[TMP5]], zeroinitializer
; CHECK-NEXT:    [[TMP7:%.*]] = mul <vscale x 2 x i7> [[TMP6]], shufflevector (<vscale x 2 x i7> insertelement (<vscale x 2 x i7> poison, i7 1, i32 0), <vscale x 2 x i7> poison, <vscale x 2 x i32> zeroinitializer)
; CHECK-NEXT:    [[INDUCTION:%.*]] = add <vscale x 2 x i7> zeroinitializer, [[TMP7]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[VEC_IND:%.*]] = phi <vscale x 2 x i7> [ [[INDUCTION]], %vector.ph ], [ [[VEC_IND_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[TMP10:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP11:%.*]] = add <vscale x 2 x i7> [[VEC_IND]], zeroinitializer
; CHECK-NEXT:    [[TMP12:%.*]] = getelementptr inbounds i64, i64* [[DST:%.*]], i64 [[TMP10]]
; CHECK-NEXT:    [[EXT:%.+]]  = zext <vscale x 2 x i7> [[TMP11]] to <vscale x 2 x i64>
; CHECK-NEXT:    [[TMP13:%.*]] = getelementptr inbounds i64, i64* [[TMP12]], i32 0
; CHECK-NEXT:    [[TMP14:%.*]] = bitcast i64* [[TMP13]] to <vscale x 2 x i64>*
; CHECK-NEXT:    store <vscale x 2 x i64> [[EXT]], <vscale x 2 x i64>* [[TMP14]], align 8
; CHECK-NEXT:    [[TMP15:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP16:%.*]] = mul i64 [[TMP15]], 2
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], [[TMP16]]
; CHECK-NEXT:    [[VEC_IND_NEXT]] = add <vscale x 2 x i7> [[VEC_IND]],
;
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv1294 = phi i7 [ %indvars.iv.next1295, %for.body ], [ 0, %entry ]
  %indvars.iv1286 = phi i64 [ %indvars.iv.next1287, %for.body ], [ 0, %entry ]
  %addi7 = add i7 %indvars.iv1294, 0
  %arrayidx = getelementptr inbounds i64, i64* %dst, i64 %indvars.iv1286
  %ext = zext i7 %addi7 to i64
  store i64 %ext, i64* %arrayidx, align 8
  %indvars.iv.next1287 = add nuw nsw i64 %indvars.iv1286, 1
  %indvars.iv.next1295 = add i7 %indvars.iv1294, 1
  %exitcond = icmp eq i64 %indvars.iv.next1287, 64
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}


; DEBUG: Found an estimated cost of Invalid for VF vscale x 1 For instruction:   %indvars.iv1294 = phi i3 [ %indvars.iv.next1295, %for.body ], [ 0, %entry ]
; DEBUG: Found an estimated cost of Invalid for VF vscale x 1 For instruction:   %zexti3 = zext i3 %indvars.iv1294 to i64
; DEBUG: Found an estimated cost of Invalid for VF vscale x 1 For instruction:   %indvars.iv.next1295 = add i3 %indvars.iv1294, 1

define void @induction_i3_zext(i64* %dst) #0 {
; CHECK-LABEL: @induction_i3_zext(
; CHECK:       vector.ph:
; CHECK:         [[TMP4:%.*]] = call <vscale x 2 x i8> @llvm.experimental.stepvector.nxv2i8()
; CHECK:         [[TMP5:%.*]] = trunc <vscale x 2 x i8> [[TMP4]] to <vscale x 2 x i3>
; CHECK-NEXT:    [[TMP6:%.*]] = add <vscale x 2 x i3> [[TMP5]], zeroinitializer
; CHECK-NEXT:    [[TMP7:%.*]] = mul <vscale x 2 x i3> [[TMP6]], shufflevector (<vscale x 2 x i3> insertelement (<vscale x 2 x i3> poison, i3 1, i32 0), <vscale x 2 x i3> poison, <vscale x 2 x i32> zeroinitializer)
; CHECK-NEXT:    [[INDUCTION:%.*]] = add <vscale x 2 x i3> zeroinitializer, [[TMP7]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[VEC_IND:%.*]] = phi <vscale x 2 x i3> [ [[INDUCTION]], %vector.ph ], [ [[VEC_IND_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[TMP9:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP10:%.*]] = zext <vscale x 2 x i3> [[VEC_IND]] to <vscale x 2 x i64>
; CHECK-NEXT:    [[TMP12:%.*]] = getelementptr inbounds i64, i64* [[DST:%.*]], i64 [[TMP9]]
; CHECK-NEXT:    [[TMP13:%.*]] = getelementptr inbounds i64, i64* [[TMP12]], i32 0
; CHECK-NEXT:    [[TMP14:%.*]] = bitcast i64* [[TMP13]] to <vscale x 2 x i64>*
; CHECK-NEXT:    store <vscale x 2 x i64> [[TMP10]], <vscale x 2 x i64>* [[TMP14]], align 8
; CHECK-NEXT:    [[TMP15:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP16:%.*]] = mul i64 [[TMP15]], 2
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], [[TMP16]]
; CHECK-NEXT:    [[VEC_IND_NEXT]] = add <vscale x 2 x i3> [[VEC_IND]],
;
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv1294 = phi i3 [ %indvars.iv.next1295, %for.body ], [ 0, %entry ]
  %indvars.iv1286 = phi i64 [ %indvars.iv.next1287, %for.body ], [ 0, %entry ]
  %zexti3 = zext i3 %indvars.iv1294 to i64
  %arrayidx = getelementptr inbounds i64, i64* %dst, i64 %indvars.iv1286
  store i64 %zexti3, i64* %arrayidx, align 8
  %indvars.iv.next1287 = add nuw nsw i64 %indvars.iv1286, 1
  %indvars.iv.next1295 = add i3 %indvars.iv1294, 1
  %exitcond = icmp eq i64 %indvars.iv.next1287, 64
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}


attributes #0 = {"target-features"="+sve"}
