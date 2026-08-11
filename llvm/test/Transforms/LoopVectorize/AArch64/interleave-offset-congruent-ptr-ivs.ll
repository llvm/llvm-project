; RUN: opt -passes='loop-mssa(indvars),loop-vectorize' -force-vector-interleave=1 -S %s | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

define void @nested_interleaved_store(ptr noalias %src, ptr noalias %dst, i32 %width, i32 %height, i32 %srcStride) {
; CHECK-LABEL: define void @nested_interleaved_store(
; CHECK-SAME: ptr noalias [[SRC:%.*]], ptr noalias [[DST:%.*]], i32 [[WIDTH:%.*]], i32 [[HEIGHT:%.*]], i32 [[SRCSTRIDE:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*]]:
; CHECK-NEXT:    [[UMAX:%.*]] = call i32 @llvm.umax.i32(i32 [[WIDTH]], i32 1)
; CHECK-NEXT:    [[WIDE_TRIP_COUNT5:%.*]] = zext i32 [[HEIGHT]] to i64
; CHECK-NEXT:    [[TMP0:%.*]] = add i32 [[UMAX]], -1
; CHECK-NEXT:    [[TMP1:%.*]] = zext i32 [[TMP0]] to i64
; CHECK-NEXT:    [[TMP2:%.*]] = shl nuw nsw i64 [[TMP1]], 3
; CHECK-NEXT:    [[TMP3:%.*]] = add nuw nsw i64 [[TMP2]], 8
; CHECK-NEXT:    [[TMP4:%.*]] = zext i32 [[SRCSTRIDE]] to i64
; CHECK-NEXT:    [[TMP5:%.*]] = shl nuw nsw i64 [[TMP1]], 2
; CHECK-NEXT:    [[TMP6:%.*]] = add nuw nsw i64 [[TMP5]], 4
; CHECK-NEXT:    [[SCEVGEP8:%.*]] = getelementptr i8, ptr [[SRC]], i64 [[TMP6]]
; CHECK-NEXT:    br label %[[OUTER_HEADER:.*]]
; CHECK:       [[OUTER_HEADER]]:
; CHECK-NEXT:    [[INDVARS_IV2:%.*]] = phi i64 [ [[INDVARS_IV_NEXT3:%.*]], %[[OUTER_LATCH:.*]] ], [ 0, %[[ENTRY]] ]
; CHECK-NEXT:    [[PA_OUTER:%.*]] = phi ptr [ [[DST]], %[[ENTRY]] ], [ [[PA_LCSSA:%.*]], %[[OUTER_LATCH]] ]
; CHECK-NEXT:    [[TMP7:%.*]] = mul i64 [[TMP4]], [[INDVARS_IV2]]
; CHECK-NEXT:    [[TMP8:%.*]] = trunc i64 [[TMP7]] to i32
; CHECK-NEXT:    [[TMP9:%.*]] = zext i32 [[TMP8]] to i64
; CHECK-NEXT:    [[TMP10:%.*]] = shl nuw nsw i64 [[TMP9]], 2
; CHECK-NEXT:    [[SCEVGEP7:%.*]] = getelementptr i8, ptr [[SRC]], i64 [[TMP10]]
; CHECK-NEXT:    [[SCEVGEP9:%.*]] = getelementptr i8, ptr [[SCEVGEP8]], i64 [[TMP10]]
; CHECK-NEXT:    [[EXITCOND6:%.*]] = icmp ne i64 [[INDVARS_IV2]], [[WIDE_TRIP_COUNT5]]
; CHECK-NEXT:    br i1 [[EXITCOND6]], label %[[INNER_PREHEADER:.*]], label %[[EXIT:.*]]
; CHECK:       [[INNER_PREHEADER]]:
; CHECK-NEXT:    [[TMP11:%.*]] = trunc nuw i64 [[INDVARS_IV2]] to i32
; CHECK-NEXT:    [[ROWOFF:%.*]] = mul i32 [[TMP11]], [[SRCSTRIDE]]
; CHECK-NEXT:    [[ROWOFF_EXT:%.*]] = zext i32 [[ROWOFF]] to i64
; CHECK-NEXT:    [[ROW:%.*]] = getelementptr inbounds nuw float, ptr [[SRC]], i64 [[ROWOFF_EXT]]
; CHECK-NEXT:    [[WIDE_TRIP_COUNT:%.*]] = zext i32 [[UMAX]] to i64
; CHECK-NEXT:    [[MIN_ITERS_CHECK:%.*]] = icmp ult i64 [[WIDE_TRIP_COUNT]], 4
; CHECK-NEXT:    br i1 [[MIN_ITERS_CHECK]], label %[[SCALAR_PH:.*]], label %[[VECTOR_MEMCHECK:.*]]
; CHECK:       [[VECTOR_MEMCHECK]]:
; CHECK-NEXT:    [[SCEVGEP:%.*]] = getelementptr i8, ptr [[PA_OUTER]], i64 [[TMP3]]
; CHECK-NEXT:    [[BOUND0:%.*]] = icmp ult ptr [[PA_OUTER]], [[SCEVGEP9]]
; CHECK-NEXT:    [[BOUND1:%.*]] = icmp ult ptr [[SCEVGEP7]], [[SCEVGEP]]
; CHECK-NEXT:    [[FOUND_CONFLICT:%.*]] = and i1 [[BOUND0]], [[BOUND1]]
; CHECK-NEXT:    br i1 [[FOUND_CONFLICT]], label %[[SCALAR_PH]], label %[[VECTOR_PH:.*]]
; CHECK:       [[VECTOR_PH]]:
; CHECK-NEXT:    [[N_MOD_VF:%.*]] = urem i64 [[WIDE_TRIP_COUNT]], 4
; CHECK-NEXT:    [[N_VEC:%.*]] = sub i64 [[WIDE_TRIP_COUNT]], [[N_MOD_VF]]
; CHECK-NEXT:    [[TMP12:%.*]] = shl i64 [[N_VEC]], 3
; CHECK-NEXT:    [[TMP13:%.*]] = getelementptr i8, ptr [[PA_OUTER]], i64 [[TMP12]]
; CHECK-NEXT:    br label %[[VECTOR_BODY:.*]]
; CHECK:       [[VECTOR_BODY]]:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %[[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], %[[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP14:%.*]] = shl i64 [[INDEX]], 3
; CHECK-NEXT:    [[NEXT_GEP:%.*]] = getelementptr i8, ptr [[PA_OUTER]], i64 [[TMP14]]
; CHECK-NEXT:    [[TMP15:%.*]] = getelementptr inbounds nuw float, ptr [[ROW]], i64 [[INDEX]]
; CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <4 x float>, ptr [[TMP15]], align 4, !alias.scope [[META0:![0-9]+]]
; CHECK-NEXT:    [[TMP16:%.*]] = fmul <4 x float> [[WIDE_LOAD]], splat (float 2.000000e+00)
; CHECK-NEXT:    [[TMP17:%.*]] = fmul <4 x float> [[WIDE_LOAD]], splat (float 3.000000e+00)
; CHECK-NEXT:    [[TMP18:%.*]] = shufflevector <4 x float> [[TMP16]], <4 x float> [[TMP17]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; CHECK-NEXT:    [[INTERLEAVED_VEC:%.*]] = shufflevector <8 x float> [[TMP18]], <8 x float> poison, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
; CHECK-NEXT:    store <8 x float> [[INTERLEAVED_VEC]], ptr [[NEXT_GEP]], align 4, !alias.scope [[META3:![0-9]+]], !noalias [[META0]]
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 4
; CHECK-NEXT:    [[TMP19:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[TMP19]], label %[[MIDDLE_BLOCK:.*]], label %[[VECTOR_BODY]], !llvm.loop [[LOOP5:![0-9]+]]
; CHECK:       [[MIDDLE_BLOCK]]:
; CHECK-NEXT:    [[CMP_N:%.*]] = icmp eq i64 [[WIDE_TRIP_COUNT]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[CMP_N]], label %[[INNER_EXIT:.*]], label %[[SCALAR_PH]]
; CHECK:       [[SCALAR_PH]]:
; CHECK-NEXT:    [[BC_RESUME_VAL:%.*]] = phi i64 [ [[N_VEC]], %[[MIDDLE_BLOCK]] ], [ 0, %[[INNER_PREHEADER]] ], [ 0, %[[VECTOR_MEMCHECK]] ]
; CHECK-NEXT:    [[BC_RESUME_VAL10:%.*]] = phi ptr [ [[TMP13]], %[[MIDDLE_BLOCK]] ], [ [[PA_OUTER]], %[[INNER_PREHEADER]] ], [ [[PA_OUTER]], %[[VECTOR_MEMCHECK]] ]
; CHECK-NEXT:    br label %[[INNER_HEADER:.*]]
; CHECK:       [[INNER_HEADER]]:
; CHECK-NEXT:    [[INDVARS_IV:%.*]] = phi i64 [ [[INDVARS_IV_NEXT:%.*]], %[[INNER_HEADER]] ], [ [[BC_RESUME_VAL]], %[[SCALAR_PH]] ]
; CHECK-NEXT:    [[PA_INNER:%.*]] = phi ptr [ [[BC_RESUME_VAL10]], %[[SCALAR_PH]] ], [ [[PA_NEXT:%.*]], %[[INNER_HEADER]] ]
; CHECK-NEXT:    [[PB_INNER_OFF:%.*]] = getelementptr i8, ptr [[PA_INNER]], i64 4
; CHECK-NEXT:    [[ELT_ADDR:%.*]] = getelementptr inbounds nuw float, ptr [[ROW]], i64 [[INDVARS_IV]]
; CHECK-NEXT:    [[ELT:%.*]] = load float, ptr [[ELT_ADDR]], align 4
; CHECK-NEXT:    [[MUL2:%.*]] = fmul float [[ELT]], 2.000000e+00
; CHECK-NEXT:    store float [[MUL2]], ptr [[PA_INNER]], align 4
; CHECK-NEXT:    [[MUL3:%.*]] = fmul float [[ELT]], 3.000000e+00
; CHECK-NEXT:    store float [[MUL3]], ptr [[PB_INNER_OFF]], align 4
; CHECK-NEXT:    [[PA_NEXT]] = getelementptr inbounds float, ptr [[PA_INNER]], i64 2
; CHECK-NEXT:    [[INDVARS_IV_NEXT]] = add nuw nsw i64 [[INDVARS_IV]], 1
; CHECK-NEXT:    [[EXITCOND:%.*]] = icmp ne i64 [[INDVARS_IV_NEXT]], [[WIDE_TRIP_COUNT]]
; CHECK-NEXT:    br i1 [[EXITCOND]], label %[[INNER_HEADER]], label %[[INNER_EXIT]], !llvm.loop [[LOOP8:![0-9]+]]
; CHECK:       [[INNER_EXIT]]:
; CHECK-NEXT:    [[PA_LCSSA]] = phi ptr [ [[PA_NEXT]], %[[INNER_HEADER]] ], [ [[TMP13]], %[[MIDDLE_BLOCK]] ]
; CHECK-NEXT:    br label %[[OUTER_LATCH]]
; CHECK:       [[OUTER_LATCH]]:
; CHECK-NEXT:    [[INDVARS_IV_NEXT3]] = add nuw nsw i64 [[INDVARS_IV2]], 1
; CHECK-NEXT:    br label %[[OUTER_HEADER]]
; CHECK:       [[EXIT]]:
; CHECK-NEXT:    ret void
;
entry:
  %pB.start = getelementptr inbounds float, ptr %dst, i64 1
  br label %outer.header

outer.header:
  %pA.outer = phi ptr [ %dst, %entry ], [ %pA.lcssa, %outer.latch ]
  %pB.outer = phi ptr [ %pB.start, %entry ], [ %pB.lcssa, %outer.latch ]
  %y = phi i32 [ 0, %entry ], [ %y.next, %outer.latch ]
  %outer.cmp = icmp ult i32 %y, %height
  br i1 %outer.cmp, label %inner.preheader, label %exit

inner.preheader:
  %rowoff = mul i32 %y, %srcStride
  %rowoff.ext = zext i32 %rowoff to i64
  %row = getelementptr inbounds nuw float, ptr %src, i64 %rowoff.ext
  br label %inner.header

inner.header:
  %pA.inner = phi ptr [ %pA.outer, %inner.preheader ], [ %pA.next, %inner.header ]
  %pB.inner = phi ptr [ %pB.outer, %inner.preheader ], [ %pB.next, %inner.header ]
  %x = phi i32 [ 0, %inner.preheader ], [ %x.next, %inner.header ]
  %x.ext = zext i32 %x to i64
  %elt.addr = getelementptr inbounds nuw float, ptr %row, i64 %x.ext
  %elt = load float, ptr %elt.addr, align 4
  %mul2 = fmul float %elt, 2.000000e+00
  store float %mul2, ptr %pA.inner, align 4
  %mul3 = fmul float %elt, 3.000000e+00
  store float %mul3, ptr %pB.inner, align 4
  %pA.next = getelementptr inbounds float, ptr %pA.inner, i64 2
  %pB.next = getelementptr inbounds float, ptr %pB.inner, i64 2
  %x.next = add nuw i32 %x, 1
  %inner.cmp = icmp ult i32 %x.next, %width
  br i1 %inner.cmp, label %inner.header, label %inner.exit

inner.exit:
  %pA.lcssa = phi ptr [ %pA.next, %inner.header ]
  %pB.lcssa = phi ptr [ %pB.next, %inner.header ]
  br label %outer.latch

outer.latch:
  %y.next = add nuw i32 %y, 1
  br label %outer.header

exit:
  ret void
}
;.
; CHECK: [[META0]] = !{[[META1:![0-9]+]]}
; CHECK: [[META1]] = distinct !{[[META1]], [[META2:![0-9]+]]}
; CHECK: [[META2]] = distinct !{[[META2]], !"LVerDomain"}
; CHECK: [[META3]] = !{[[META4:![0-9]+]]}
; CHECK: [[META4]] = distinct !{[[META4]], [[META2]]}
; CHECK: [[LOOP5]] = distinct !{[[LOOP5]], [[META6:![0-9]+]], [[META7:![0-9]+]]}
; CHECK: [[META6]] = !{!"llvm.loop.isvectorized", i32 1}
; CHECK: [[META7]] = !{!"llvm.loop.unroll.runtime.disable"}
; CHECK: [[LOOP8]] = distinct !{[[LOOP8]], [[META6]]}
;.
