; RUN: opt < %s -passes=loop-vectorize -S | FileCheck %s
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: uwtable vscale_range(1,16)
define dso_local void @foo(i32 %arg, ptr noalias %arg1, ptr noalias %arg2, ptr noalias readonly %arg3, ptr noalias readonly %arg4, ptr noalias readonly %arg5, i8 %arg6) local_unnamed_addr #0 {
; CHECK-LABEL: define dso_local void @foo(
; CHECK-SAME: i32 [[ARG:%.*]], ptr noalias [[ARG1:%.*]], ptr noalias [[ARG2:%.*]], ptr noalias readonly [[ARG3:%.*]], ptr noalias readonly [[ARG4:%.*]], ptr noalias readonly [[ARG5:%.*]], i8 [[ARG6:%.*]]) local_unnamed_addr #[[ATTR0:[0-9]+]] {
; CHECK-NEXT:  [[BB:.*:]]
; CHECK-NEXT:    [[ICMP:%.*]] = icmp sgt i32 [[ARG]], 0
; CHECK-NEXT:    br i1 [[ICMP]], label %[[BB7:.*]], label %[[BB25:.*]]
; CHECK:       [[BB7]]:
; CHECK-NEXT:    [[ZEXT:%.*]] = zext nneg i32 [[ARG]] to i64
; CHECK-NEXT:    [[TMP0:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP1:%.*]] = shl nuw nsw i64 [[TMP0]], 4
; CHECK-NEXT:    [[MIN_ITERS_CHECK:%.*]] = icmp ult i64 [[ZEXT]], [[TMP1]]
; CHECK-NEXT:    br i1 [[MIN_ITERS_CHECK]], label %[[SCALAR_PH:.*]], label %[[VECTOR_PH:.*]]
; CHECK:       [[VECTOR_PH]]:
; CHECK-NEXT:    [[TMP2:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP3:%.*]] = mul nuw i64 [[TMP2]], 16
; CHECK-NEXT:    [[N_MOD_VF:%.*]] = urem i64 [[ZEXT]], [[TMP3]]
; CHECK-NEXT:    [[N_VEC:%.*]] = sub i64 [[ZEXT]], [[N_MOD_VF]]
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <vscale x 16 x i8> poison, i8 [[ARG6]], i64 0
; CHECK-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <vscale x 16 x i8> [[BROADCAST_SPLATINSERT]], <vscale x 16 x i8> poison, <vscale x 16 x i32> zeroinitializer
; CHECK-NEXT:    br label %[[VECTOR_BODY:.*]]
; CHECK:       [[VECTOR_BODY]]:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %[[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], %[[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP4:%.*]] = getelementptr inbounds nuw i8, ptr [[ARG5]], i64 [[INDEX]]
; CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <vscale x 16 x i8>, ptr [[TMP4]], align 1
; CHECK-NEXT:    [[TMP5:%.*]] = icmp uge <vscale x 16 x i8> [[WIDE_LOAD]], [[BROADCAST_SPLAT]]
; CHECK-NEXT:    [[TMP6:%.*]] = getelementptr i8, ptr [[ARG1]], i64 [[INDEX]]
; CHECK-NEXT:    [[WIDE_MASKED_LOAD:%.*]] = call <vscale x 16 x i8> @llvm.masked.load.nxv16i8.p0(ptr [[TMP6]], i32 1, <vscale x 16 x i1> [[TMP5]], <vscale x 16 x i8> poison)
; CHECK-NEXT:    [[TMP7:%.*]] = getelementptr i8, ptr [[ARG3]], i64 [[INDEX]]
; CHECK-NEXT:    [[WIDE_MASKED_LOAD1:%.*]] = call <vscale x 16 x i8> @llvm.masked.load.nxv16i8.p0(ptr [[TMP7]], i32 1, <vscale x 16 x i1> [[TMP5]], <vscale x 16 x i8> poison)
; CHECK-NEXT:    [[TMP8:%.*]] = getelementptr i8, ptr [[ARG4]], i64 [[INDEX]]
; CHECK-NEXT:    [[WIDE_MASKED_LOAD2:%.*]] = call <vscale x 16 x i8> @llvm.masked.load.nxv16i8.p0(ptr [[TMP8]], i32 1, <vscale x 16 x i1> [[TMP5]], <vscale x 16 x i8> poison)
; CHECK-NEXT:    [[TMP9:%.*]] = mul <vscale x 16 x i8> [[WIDE_MASKED_LOAD2]], [[WIDE_MASKED_LOAD1]]
; CHECK-NEXT:    [[TMP10:%.*]] = add <vscale x 16 x i8> [[TMP9]], [[WIDE_MASKED_LOAD]]
; CHECK-NEXT:    call void @llvm.masked.store.nxv16i8.p0(<vscale x 16 x i8> [[TMP10]], ptr [[TMP6]], i32 1, <vscale x 16 x i1> [[TMP5]])
; CHECK-NEXT:    [[TMP11:%.*]] = getelementptr i8, ptr [[ARG2]], i64 [[INDEX]]
; CHECK-NEXT:    [[WIDE_MASKED_LOAD3:%.*]] = call <vscale x 16 x i8> @llvm.masked.load.nxv16i8.p0(ptr [[TMP11]], i32 1, <vscale x 16 x i1> [[TMP5]], <vscale x 16 x i8> poison)
; CHECK-NEXT:    [[TMP12:%.*]] = mul <vscale x 16 x i8> [[WIDE_MASKED_LOAD1]], [[WIDE_MASKED_LOAD1]]
; CHECK-NEXT:    [[TMP13:%.*]] = add <vscale x 16 x i8> [[WIDE_MASKED_LOAD3]], [[TMP12]]
; CHECK-NEXT:    call void @llvm.masked.store.nxv16i8.p0(<vscale x 16 x i8> [[TMP13]], ptr [[TMP11]], i32 1, <vscale x 16 x i1> [[TMP5]])
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], [[TMP3]]
; CHECK-NEXT:    [[TMP14:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[TMP14]], label %[[MIDDLE_BLOCK:.*]], label %[[VECTOR_BODY]], !llvm.loop [[LOOP0:![0-9]+]]
; CHECK:       [[MIDDLE_BLOCK]]:
; CHECK-NEXT:    [[CMP_N:%.*]] = icmp eq i64 [[ZEXT]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[CMP_N]], label %[[BB24:.*]], label %[[SCALAR_PH]]
; CHECK:       [[SCALAR_PH]]:
; CHECK-NEXT:    [[BC_RESUME_VAL:%.*]] = phi i64 [ [[N_VEC]], %[[MIDDLE_BLOCK]] ], [ 0, %[[BB7]] ]
; CHECK-NEXT:    br label %[[BB8:.*]]
; CHECK:       [[BB8]]:
; CHECK-NEXT:    [[PHI:%.*]] = phi i64 [ [[BC_RESUME_VAL]], %[[SCALAR_PH]] ], [ [[ADD22:%.*]], %[[BB21:.*]] ]
; CHECK-NEXT:    [[GETELEMENTPTR:%.*]] = getelementptr inbounds nuw i8, ptr [[ARG5]], i64 [[PHI]]
; CHECK-NEXT:    [[LOAD:%.*]] = load i8, ptr [[GETELEMENTPTR]], align 1
; CHECK-NEXT:    [[ICMP9:%.*]] = icmp ult i8 [[LOAD]], [[ARG6]]
; CHECK-NEXT:    br i1 [[ICMP9]], label %[[BB21]], label %[[BB10:.*]]
; CHECK:       [[BB10]]:
; CHECK-NEXT:    [[GETELEMENTPTR11:%.*]] = getelementptr inbounds nuw i8, ptr [[ARG1]], i64 [[PHI]]
; CHECK-NEXT:    [[LOAD12:%.*]] = load i8, ptr [[GETELEMENTPTR11]], align 1
; CHECK-NEXT:    [[GETELEMENTPTR13:%.*]] = getelementptr inbounds nuw i8, ptr [[ARG3]], i64 [[PHI]]
; CHECK-NEXT:    [[LOAD14:%.*]] = load i8, ptr [[GETELEMENTPTR13]], align 1
; CHECK-NEXT:    [[GETELEMENTPTR15:%.*]] = getelementptr inbounds nuw i8, ptr [[ARG4]], i64 [[PHI]]
; CHECK-NEXT:    [[LOAD16:%.*]] = load i8, ptr [[GETELEMENTPTR15]], align 1
; CHECK-NEXT:    [[MUL:%.*]] = mul i8 [[LOAD16]], [[LOAD14]]
; CHECK-NEXT:    [[ADD:%.*]] = add i8 [[MUL]], [[LOAD12]]
; CHECK-NEXT:    store i8 [[ADD]], ptr [[GETELEMENTPTR11]], align 1
; CHECK-NEXT:    [[GETELEMENTPTR17:%.*]] = getelementptr inbounds nuw i8, ptr [[ARG2]], i64 [[PHI]]
; CHECK-NEXT:    [[LOAD18:%.*]] = load i8, ptr [[GETELEMENTPTR17]], align 1
; CHECK-NEXT:    [[MUL19:%.*]] = mul i8 [[LOAD14]], [[LOAD14]]
; CHECK-NEXT:    [[ADD20:%.*]] = add i8 [[LOAD18]], [[MUL19]]
; CHECK-NEXT:    store i8 [[ADD20]], ptr [[GETELEMENTPTR17]], align 1
; CHECK-NEXT:    br label %[[BB21]]
; CHECK:       [[BB21]]:
; CHECK-NEXT:    [[ADD22]] = add nuw nsw i64 [[PHI]], 1
; CHECK-NEXT:    [[ICMP23:%.*]] = icmp eq i64 [[ADD22]], [[ZEXT]]
; CHECK-NEXT:    br i1 [[ICMP23]], label %[[BB24]], label %[[BB8]], !llvm.loop [[LOOP4:![0-9]+]]
; CHECK:       [[BB24]]:
; CHECK-NEXT:    br label %[[BB25]]
; CHECK:       [[BB25]]:
; CHECK-NEXT:    ret void
;
bb:
  %icmp = icmp sgt i32 %arg, 0
  br i1 %icmp, label %bb7, label %bb25

bb7:                                              ; preds = %bb
  %zext = zext nneg i32 %arg to i64
  br label %bb8

bb8:                                              ; preds = %bb21, %bb7
  %phi = phi i64 [ 0, %bb7 ], [ %add22, %bb21 ]
  %getelementptr = getelementptr inbounds nuw i8, ptr %arg5, i64 %phi
  %load = load i8, ptr %getelementptr, align 1
  %icmp9 = icmp ult i8 %load, %arg6
  br i1 %icmp9, label %bb21, label %bb10

bb10:                                             ; preds = %bb8
  %getelementptr11 = getelementptr inbounds nuw i8, ptr %arg1, i64 %phi
  %load12 = load i8, ptr %getelementptr11, align 1
  %getelementptr13 = getelementptr inbounds nuw i8, ptr %arg3, i64 %phi
  %load14 = load i8, ptr %getelementptr13, align 1
  %getelementptr15 = getelementptr inbounds nuw i8, ptr %arg4, i64 %phi
  %load16 = load i8, ptr %getelementptr15, align 1
  %mul = mul i8 %load16, %load14
  %add = add i8 %mul, %load12
  store i8 %add, ptr %getelementptr11, align 1
  %getelementptr17 = getelementptr inbounds nuw i8, ptr %arg2, i64 %phi
  %load18 = load i8, ptr %getelementptr17, align 1
  %mul19 = mul i8 %load14, %load14
  %add20 = add i8 %load18, %mul19
  store i8 %add20, ptr %getelementptr17, align 1
  br label %bb21

bb21:                                             ; preds = %bb10, %bb8
  %add22 = add nuw nsw i64 %phi, 1
  %icmp23 = icmp eq i64 %add22, %zext
  br i1 %icmp23, label %bb24, label %bb8, !llvm.loop !0

bb24:                                             ; preds = %bb21
  br label %bb25

bb25:                                             ; preds = %bb24, %bb
  ret void
}

attributes #0 = { uwtable vscale_range(1,16) "aarch64_pstate_sm_body" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+bf16,+fp-armv8,+fullfp16,+neon,+outline-atomics,+sme,+v8a,-fmv" }

!0 = distinct !{!0, !1, !2, !3, !4}
!1 = !{!"llvm.loop.mustprogress"}
!2 = !{!"llvm.loop.vectorize.width", i32 16}
!3 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
!4 = !{!"llvm.loop.vectorize.enable", i1 true}
;.
; CHECK: [[LOOP0]] = distinct !{[[LOOP0]], [[META1:![0-9]+]], [[META2:![0-9]+]], [[META3:![0-9]+]]}
; CHECK: [[META1]] = !{!"llvm.loop.mustprogress"}
; CHECK: [[META2]] = !{!"llvm.loop.isvectorized", i32 1}
; CHECK: [[META3]] = !{!"llvm.loop.unroll.runtime.disable"}
; CHECK: [[LOOP4]] = distinct !{[[LOOP4]], [[META1]], [[META3]], [[META2]]}
;.
