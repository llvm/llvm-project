; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -S < %s -debug -prefer-predicate-over-epilogue=scalar-epilogue 2>%t | FileCheck %s
; RUN: cat %t | FileCheck %s --check-prefix=DEBUG

target triple = "aarch64-unknown-linux-gnu"

; DEBUG: Cost of Invalid for VF vscale x 1: induction instruction   %indvars.iv.next1295 = add i7 %indvars.iv1294, 1
; DEBUG: Cost of Invalid for VF vscale x 1: induction instruction   %indvars.iv1294 = phi i7 [ %indvars.iv.next1295, %for.body ], [ 0, %entry ]
; DEBUG: Cost of Invalid for VF vscale x 1: WIDEN ir<%addi7> = add ir<%indvars.iv1294>, ir<0>

define void @induction_i7(ptr %dst) #0 {
; CHECK-LABEL: define void @induction_i7(
; CHECK-SAME: ptr [[DST:%.*]])
; CHECK:       vector.ph:
; CHECK-NEXT:    [[TMP4:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP40:%.*]] = mul nuw i64 [[TMP4]], 2
; CHECK-NEXT:    [[TMP5:%.*]] = mul i64 [[TMP40]], 2
; CHECK-NEXT:    [[N_MOD_VF:%.*]] = urem i64 64, [[TMP5]]
; CHECK-NEXT:    [[N_VEC:%.*]] = sub i64 64, [[N_MOD_VF]]
; CHECK-NEXT:    [[IND_END:%.*]] = trunc i64 [[N_VEC]] to i7
; CHECK-NEXT:    [[DOTSPLATINSERT:%.*]] = insertelement <vscale x 2 x i64> poison, i64 [[TMP40]], i64 0
; CHECK-NEXT:    [[DOTSPLAT_:%.*]] = shufflevector <vscale x 2 x i64> [[DOTSPLATINSERT]], <vscale x 2 x i64> poison, <vscale x 2 x i32> zeroinitializer
; CHECK-NEXT:    [[DOTSPLAT:%.*]] = trunc <vscale x 2 x i64> [[DOTSPLAT_]] to <vscale x 2 x i7>
; CHECK-NEXT:    [[TMP6:%.*]] = call <vscale x 2 x i8> @llvm.stepvector.nxv2i8()
; CHECK-NEXT:    [[TMP7:%.*]] = trunc <vscale x 2 x i8> [[TMP6]] to <vscale x 2 x i7> 
; CHECK-NEXT:    [[TMP9:%.*]] = mul <vscale x 2 x i7> [[TMP7]], splat (i7 1)
; CHECK-NEXT:    [[INDUCTION:%.*]] = add <vscale x 2 x i7> zeroinitializer, [[TMP9]]
; CHECK-NEXT:    br label %[[VECTOR_BODY:.*]]
; CHECK:       [[VECTOR_BODY]]:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %[[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_IND:%.*]] = phi <vscale x 2 x i7> [ [[INDUCTION]], %vector.ph ], [ [[VEC_IND_NEXT:%.*]], %[[VECTOR_BODY]] ]
; CHECK-NEXT:    [[STEP_ADD:%.*]] = add <vscale x 2 x i7> [[VEC_IND]], [[DOTSPLAT]]
; CHECK-NEXT:    [[TMP19:%.*]] = add <vscale x 2 x i7> [[VEC_IND]], zeroinitializer
; CHECK-NEXT:    [[TMP20:%.*]] = add <vscale x 2 x i7> [[STEP_ADD]], zeroinitializer
; CHECK-NEXT:    [[TMP21:%.*]] = getelementptr inbounds i64, ptr [[DST]], i64 [[INDEX]]
; CHECK-NEXT:    [[TMP23:%.*]] = zext <vscale x 2 x i7> [[TMP19]] to <vscale x 2 x i64>
; CHECK-NEXT:    [[TMP24:%.*]] = zext <vscale x 2 x i7> [[TMP20]] to <vscale x 2 x i64>
; CHECK-NEXT:    [[TMP26:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP27:%.*]] = mul nuw i64 [[TMP26]], 2
; CHECK-NEXT:    [[TMP28:%.*]] = getelementptr inbounds i64, ptr [[TMP21]], i64 [[TMP27]]
; CHECK-NEXT:    store <vscale x 2 x i64> [[TMP23]], ptr [[TMP21]], align 8
; CHECK-NEXT:    store <vscale x 2 x i64> [[TMP24]], ptr [[TMP28]], align 8
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], [[TMP5]]
; CHECK-NEXT:    [[VEC_IND_NEXT]] = add <vscale x 2 x i7> [[STEP_ADD]], [[DOTSPLAT]]
; CHECK-NEXT:    [[TMP29:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[TMP29]], label %middle.block, label %[[VECTOR_BODY]]
;

entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv1294 = phi i7 [ %indvars.iv.next1295, %for.body ], [ 0, %entry ]
  %indvars.iv1286 = phi i64 [ %indvars.iv.next1287, %for.body ], [ 0, %entry ]
  %addi7 = add i7 %indvars.iv1294, 0
  %arrayidx = getelementptr inbounds i64, ptr %dst, i64 %indvars.iv1286
  %ext = zext i7 %addi7 to i64
  store i64 %ext, ptr %arrayidx, align 8
  %indvars.iv.next1287 = add nuw nsw i64 %indvars.iv1286, 1
  %indvars.iv.next1295 = add i7 %indvars.iv1294, 1
  %exitcond = icmp eq i64 %indvars.iv.next1287, 64
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}


; DEBUG: Cost of Invalid for VF vscale x 1: induction instruction   %indvars.iv.next1295 = add i3 %indvars.iv1294, 1
; DEBUG: Cost of Invalid for VF vscale x 1: induction instruction   %indvars.iv1294 = phi i3 [ %indvars.iv.next1295, %for.body ], [ 0, %entry ]
; DEBUG: Cost of Invalid for VF vscale x 1: WIDEN-CAST ir<%zexti3> = zext ir<%indvars.iv1294> to i64

define void @induction_i3_zext(ptr %dst) #0 {
; CHECK-LABEL: define void @induction_i3_zext(
; CHECK-SAME: ptr [[DST:%.*]])
; CHECK:       vector.ph:
; CHECK-NEXT:    [[TMP4:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP40:%.*]] = mul nuw i64 [[TMP4]], 2
; CHECK-NEXT:    [[TMP5:%.*]] = mul i64 [[TMP40]], 2
; CHECK-NEXT:    [[N_MOD_VF:%.*]] = urem i64 64, [[TMP5]]
; CHECK-NEXT:    [[N_VEC:%.*]] = sub i64 64, [[N_MOD_VF]]
; CHECK-NEXT:    [[IND_END:%.*]] = trunc i64 [[N_VEC]] to i3
; CHECK-NEXT:    [[DOTSPLATINSERT:%.*]] = insertelement <vscale x 2 x i64> poison, i64 [[TMP40]], i64 0
; CHECK-NEXT:    [[DOTSPLAT_:%.*]] = shufflevector <vscale x 2 x i64> [[DOTSPLATINSERT]], <vscale x 2 x i64> poison, <vscale x 2 x i32> zeroinitializer
; CHECK-NEXT:    [[DOTSPLAT:%.*]] = trunc <vscale x 2 x i64> [[DOTSPLAT_]] to <vscale x 2 x i3>
; CHECK-NEXT:    [[TMP6:%.*]] = call <vscale x 2 x i8> @llvm.stepvector.nxv2i8()
; CHECK-NEXT:    [[TMP7:%.*]] = trunc <vscale x 2 x i8> [[TMP6]] to <vscale x 2 x i3>
; CHECK-NEXT:    [[TMP9:%.*]] = mul <vscale x 2 x i3> [[TMP7]], splat (i3 1)
; CHECK-NEXT:    [[INDUCTION:%.*]] = add <vscale x 2 x i3> zeroinitializer, [[TMP9]]
; CHECK-NEXT:    br label %[[VECTOR_BODY:.*]]
; CHECK:       [[VECTOR_BODY]]:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %[[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_IND:%.*]] = phi <vscale x 2 x i3> [ [[INDUCTION]], %vector.ph ], [ [[VEC_IND_NEXT:%.*]], %[[VECTOR_BODY]] ]
; CHECK-NEXT:    [[STEP_ADD:%.*]] = add <vscale x 2 x i3> [[VEC_IND]], [[DOTSPLAT]]
; CHECK-NEXT:    [[TMP19:%.*]] = zext <vscale x 2 x i3> [[VEC_IND]] to <vscale x 2 x i64>
; CHECK-NEXT:    [[TMP20:%.*]] = zext <vscale x 2 x i3> [[STEP_ADD]] to <vscale x 2 x i64>
; CHECK-NEXT:    [[TMP21:%.*]] = getelementptr inbounds i64, ptr [[DST]], i64 [[INDEX]]
; CHECK-NEXT:    [[TMP24:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP25:%.*]] = mul nuw i64 [[TMP24]], 2
; CHECK-NEXT:    [[TMP26:%.*]] = getelementptr inbounds i64, ptr [[TMP21]], i64 [[TMP25]]
; CHECK-NEXT:    store <vscale x 2 x i64> [[TMP19]], ptr [[TMP21]], align 8
; CHECK-NEXT:    store <vscale x 2 x i64> [[TMP20]], ptr [[TMP26]], align 8
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], [[TMP5]]
; CHECK-NEXT:    [[VEC_IND_NEXT]] = add <vscale x 2 x i3> [[STEP_ADD]], [[DOTSPLAT]]
; CHECK-NEXT:    [[TMP27:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[TMP27]], label %middle.block, label %[[VECTOR_BODY]]
;
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv1294 = phi i3 [ %indvars.iv.next1295, %for.body ], [ 0, %entry ]
  %indvars.iv1286 = phi i64 [ %indvars.iv.next1287, %for.body ], [ 0, %entry ]
  %zexti3 = zext i3 %indvars.iv1294 to i64
  %arrayidx = getelementptr inbounds i64, ptr %dst, i64 %indvars.iv1286
  store i64 %zexti3, ptr %arrayidx, align 8
  %indvars.iv.next1287 = add nuw nsw i64 %indvars.iv1286, 1
  %indvars.iv.next1295 = add i3 %indvars.iv1294, 1
  %exitcond = icmp eq i64 %indvars.iv.next1287, 64
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}


attributes #0 = {"target-features"="+sve"}
