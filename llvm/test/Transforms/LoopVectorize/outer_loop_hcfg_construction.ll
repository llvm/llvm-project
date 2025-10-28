; RUN: opt -S -passes=loop-vectorize -enable-vplan-native-path < %s -S | FileCheck %s

; void test(int n, int **a)
; {
;   for (int k = 0; k < n; ++k) {
;     a[k][0] = 0;
;     #pragma clang loop vectorize_width(4)
;     for (int i = 0; i < n; ++i) {
;         for (int j = 0; j < n; ++j) {
;             a[i][j] = 2 + k;
;         }
;     }
;   }
; }
;
; Make sure VPlan HCFG is constructed when we try to vectorize non-outermost loop
;
define void @non_outermost_loop_hcfg_construction(i64 %n, ptr %a) {
; CHECK-LABEL: define void @non_outermost_loop_hcfg_construction(
; CHECK-SAME: i64 [[N:%.*]], ptr [[A:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[OUTERMOST_LOOP:%.*]]
; CHECK:       outermost.loop:
; CHECK-NEXT:    [[K:%.*]] = phi i64 [ 0, [[ENTRY:%.*]] ], [ [[K_NEXT:%.*]], [[OUTERMOST_LOOP_LATCH:%.*]] ]
; CHECK-NEXT:    [[ARRAYIDX_US:%.*]] = getelementptr inbounds ptr, ptr [[A]], i64 [[K]]
; CHECK-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[ARRAYIDX_US]], align 8
; CHECK-NEXT:    store i32 0, ptr [[TMP0]], align 4
; CHECK-NEXT:    [[TMP1:%.*]] = trunc i64 [[K]] to i32
; CHECK-NEXT:    [[TMP2:%.*]] = add i32 [[TMP1]], 2
; CHECK-NEXT:    [[MIN_ITERS_CHECK:%.*]] = icmp ult i64 [[N]], 4
; CHECK-NEXT:    br i1 [[MIN_ITERS_CHECK]], label [[SCALAR_PH:%.*]], label [[VECTOR_PH:%.*]]
; CHECK:       vector.ph:
; CHECK-NEXT:    [[N_MOD_VF:%.*]] = urem i64 [[N]], 4
; CHECK-NEXT:    [[N_VEC:%.*]] = sub i64 [[N]], [[N_MOD_VF]]
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <4 x i32> poison, i32 [[TMP2]], i64 0
; CHECK-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <4 x i32> [[BROADCAST_SPLATINSERT]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT2:%.*]] = insertelement <4 x i64> poison, i64 [[N]], i64 0
; CHECK-NEXT:    [[BROADCAST_SPLAT3:%.*]] = shufflevector <4 x i64> [[BROADCAST_SPLATINSERT2]], <4 x i64> poison, <4 x i32> zeroinitializer
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[MIDDLE_LOOP_LATCH4:%.*]] ]
; CHECK-NEXT:    [[VEC_IND:%.*]] = phi <4 x i64> [ <i64 0, i64 1, i64 2, i64 3>, [[VECTOR_PH]] ], [ [[VEC_IND_NEXT:%.*]], [[MIDDLE_LOOP_LATCH4]] ]
; CHECK-NEXT:    [[TMP3:%.*]] = getelementptr inbounds ptr, ptr [[A]], <4 x i64> [[VEC_IND]]
; CHECK-NEXT:    [[WIDE_MASKED_GATHER:%.*]] = call <4 x ptr> @llvm.masked.gather.v4p0.v4p0(<4 x ptr> align 8 [[TMP3]], <4 x i1> splat (i1 true), <4 x ptr> poison)
; CHECK-NEXT:    br label [[INNERMOST_LOOP3:%.*]]
; CHECK:       innermost.loop3:
; CHECK-NEXT:    [[VEC_PHI:%.*]] = phi <4 x i64> [ zeroinitializer, [[VECTOR_BODY]] ], [ [[TMP5:%.*]], [[INNERMOST_LOOP3]] ]
; CHECK-NEXT:    [[TMP4:%.*]] = getelementptr inbounds i32, <4 x ptr> [[WIDE_MASKED_GATHER]], <4 x i64> [[VEC_PHI]]
; CHECK-NEXT:    call void @llvm.masked.scatter.v4i32.v4p0(<4 x i32> [[BROADCAST_SPLAT]], <4 x ptr> align 4 [[TMP4]], <4 x i1> splat (i1 true))
; CHECK-NEXT:    [[TMP5]] = add nuw nsw <4 x i64> [[VEC_PHI]], splat (i64 1)
; CHECK-NEXT:    [[TMP6:%.*]] = icmp eq <4 x i64> [[TMP5]], [[BROADCAST_SPLAT3]]
; CHECK-NEXT:    [[TMP7:%.*]] = extractelement <4 x i1> [[TMP6]], i32 0
; CHECK-NEXT:    br i1 [[TMP7]], label [[MIDDLE_LOOP_LATCH4]], label [[INNERMOST_LOOP3]]
; CHECK:       vector.latch:
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 4
; CHECK-NEXT:    [[VEC_IND_NEXT]] = add <4 x i64> [[VEC_IND]], splat (i64 4)
; CHECK-NEXT:    [[TMP10:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[TMP10]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP0:![0-9]+]]
; CHECK:       middle.block:
; CHECK-NEXT:    [[CMP_N:%.*]] = icmp eq i64 [[N]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[CMP_N]], label [[OUTERMOST_LOOP_LATCH]], label [[SCALAR_PH]]
; CHECK:       scalar.ph:
; CHECK-NEXT:    [[BC_RESUME_VAL:%.*]] = phi i64 [ [[N_VEC]], [[MIDDLE_BLOCK]] ], [ 0, [[OUTERMOST_LOOP]] ]
; CHECK-NEXT:    br label [[MIDDLE_LOOP:%.*]]
; CHECK:       middle.loop:
; CHECK-NEXT:    [[I:%.*]] = phi i64 [ [[I_NEXT:%.*]], [[MIDDLE_LOOP_LATCH:%.*]] ], [ [[BC_RESUME_VAL]], [[SCALAR_PH]] ]
; CHECK-NEXT:    [[ARRAYIDX11_US_US:%.*]] = getelementptr inbounds ptr, ptr [[A]], i64 [[I]]
; CHECK-NEXT:    [[TMP11:%.*]] = load ptr, ptr [[ARRAYIDX11_US_US]], align 8
; CHECK-NEXT:    br label [[INNERMOST_LOOP:%.*]]
; CHECK:       innermost.loop:
; CHECK-NEXT:    [[J:%.*]] = phi i64 [ [[J_NEXT:%.*]], [[INNERMOST_LOOP]] ], [ 0, [[MIDDLE_LOOP]] ]
; CHECK-NEXT:    [[ARRAYIDX13_US_US:%.*]] = getelementptr inbounds i32, ptr [[TMP11]], i64 [[J]]
; CHECK-NEXT:    store i32 [[TMP2]], ptr [[ARRAYIDX13_US_US]], align 4
; CHECK-NEXT:    [[J_NEXT]] = add nuw nsw i64 [[J]], 1
; CHECK-NEXT:    [[EXITCOND_NOT:%.*]] = icmp eq i64 [[J_NEXT]], [[N]]
; CHECK-NEXT:    br i1 [[EXITCOND_NOT]], label [[MIDDLE_LOOP_LATCH]], label [[INNERMOST_LOOP]]
; CHECK:       middle.loop.latch:
; CHECK-NEXT:    [[I_NEXT]] = add nuw nsw i64 [[I]], 1
; CHECK-NEXT:    [[EXITCOND41_NOT:%.*]] = icmp eq i64 [[I_NEXT]], [[N]]
; CHECK-NEXT:    br i1 [[EXITCOND41_NOT]], label [[OUTERMOST_LOOP_LATCH]], label [[MIDDLE_LOOP]], !llvm.loop [[LOOP3:![0-9]+]]
; CHECK:       outermost.loop.latch:
; CHECK-NEXT:    [[K_NEXT]] = add nuw nsw i64 [[K]], 1
; CHECK-NEXT:    [[EXITCOND47_NOT:%.*]] = icmp eq i64 [[K_NEXT]], [[N]]
; CHECK-NEXT:    br i1 [[EXITCOND47_NOT]], label [[OUTERMOST_LOOP_POSTEXIT:%.*]], label [[OUTERMOST_LOOP]]
; CHECK:       outermost.loop.postexit:
; CHECK-NEXT:    br label [[FOR_COND_CLEANUP:%.*]]
; CHECK:       for.cond.cleanup:
; CHECK-NEXT:    ret void
;
entry:
  br label %outermost.loop

outermost.loop:
  %k = phi i64 [ 0, %entry ], [ %k.next, %outermost.loop.latch ]
  %arrayidx.us = getelementptr inbounds ptr, ptr %a, i64 %k
  %0 = load ptr, ptr %arrayidx.us, align 8
  store i32 0, ptr %0, align 4
  %1 = trunc i64 %k to i32
  %2 = add i32 %1, 2
  br label %middle.loop

middle.loop:
  %i = phi i64 [ %i.next, %middle.loop.latch ], [ 0, %outermost.loop ]
  %arrayidx11.us.us = getelementptr inbounds ptr, ptr %a, i64 %i
  %3 = load ptr, ptr %arrayidx11.us.us, align 8
  br label %innermost.loop

innermost.loop:
  %j = phi i64 [ %j.next, %innermost.loop ], [ 0, %middle.loop ]
  %arrayidx13.us.us = getelementptr inbounds i32, ptr %3, i64 %j
  store i32 %2, ptr %arrayidx13.us.us, align 4
  %j.next = add nuw nsw i64 %j, 1
  %exitcond.not = icmp eq i64 %j.next, %n
  br i1 %exitcond.not, label %middle.loop.latch, label %innermost.loop

middle.loop.latch:
  %i.next = add nuw nsw i64 %i, 1
  %exitcond41.not = icmp eq i64 %i.next, %n
  br i1 %exitcond41.not, label %outermost.loop.latch, label %middle.loop, !llvm.loop !3

outermost.loop.latch:
  %k.next = add nuw nsw i64 %k, 1
  %exitcond47.not = icmp eq i64 %k.next, %n
  br i1 %exitcond47.not, label %outermost.loop.postexit, label %outermost.loop

outermost.loop.postexit:
  br label %for.cond.cleanup

for.cond.cleanup:
  ret void
}

; void non_outermost_loop_hcfg_construction_other_loops_at_same_level(long n, int **a)
; {
;   for (long k = 0; k < n; ++k) {
;     a[k][0] = 0;
;     for (long  i = 0; i < n; ++i) {
;         #pragma clang loop vectorize_width(4)
;         for (long j0 = 0; j0 < n; ++j0) {
;             for (long x = 0; x < n; ++x) {
;               a[x+i][j0] = 2 + k+x;
;             }
;         }
;
;         for (long j1 = n; j1 > 0; --j1) {
;           a[i][j1] *= j1 & 1;
;         }
;     }
;   }
; }
;
; Make sure VPlan HCFG is constructed when we try to vectorize loop with other loops at level > 0
;
define void @non_outermost_loop_hcfg_construction_other_loops_at_same_level(i64 %n, ptr %a) {
; CHECK-LABEL: define void @non_outermost_loop_hcfg_construction_other_loops_at_same_level(
; CHECK-SAME: i64 [[N:%.*]], ptr [[A:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[OUTERMOST_LOOP_K:%.*]]
; CHECK:       return:
; CHECK-NEXT:    ret void
; CHECK:       outermost.loop.k:
; CHECK-NEXT:    [[K:%.*]] = phi i64 [ [[K_NEXT:%.*]], [[OUTERMOST_LOOP_K_CLEANUP:%.*]] ], [ 0, [[ENTRY:%.*]] ]
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds ptr, ptr [[A]], i64 [[K]]
; CHECK-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[ARRAYIDX]], align 8
; CHECK-NEXT:    store i32 0, ptr [[TMP0]], align 4
; CHECK-NEXT:    [[ADD:%.*]] = add nuw nsw i64 [[K]], 2
; CHECK-NEXT:    br label [[MIDDLE_LOOP_I:%.*]]
; CHECK:       middle.loop.i:
; CHECK-NEXT:    [[I:%.*]] = phi i64 [ 0, [[OUTERMOST_LOOP_K]] ], [ [[I_NEXT:%.*]], [[MIDDLE_LOOP_I_CLEANUP:%.*]] ]
; CHECK-NEXT:    [[INVARIANT_GEP:%.*]] = getelementptr ptr, ptr [[A]], i64 [[I]]
; CHECK-NEXT:    [[MIN_ITERS_CHECK:%.*]] = icmp ult i64 [[N]], 4
; CHECK-NEXT:    br i1 [[MIN_ITERS_CHECK]], label [[SCALAR_PH:%.*]], label [[VECTOR_PH:%.*]]
; CHECK:       vector.ph:
; CHECK-NEXT:    [[N_MOD_VF:%.*]] = urem i64 [[N]], 4
; CHECK-NEXT:    [[N_VEC:%.*]] = sub i64 [[N]], [[N_MOD_VF]]
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <4 x i64> poison, i64 [[ADD]], i64 0
; CHECK-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <4 x i64> [[BROADCAST_SPLATINSERT]], <4 x i64> poison, <4 x i32> zeroinitializer
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT2:%.*]] = insertelement <4 x i64> poison, i64 [[N]], i64 0
; CHECK-NEXT:    [[BROADCAST_SPLAT3:%.*]] = shufflevector <4 x i64> [[BROADCAST_SPLATINSERT2]], <4 x i64> poison, <4 x i32> zeroinitializer
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[MIDDLE_LOOP_J0_CLEANUP4:%.*]] ]
; CHECK-NEXT:    [[VEC_IND:%.*]] = phi <4 x i64> [ <i64 0, i64 1, i64 2, i64 3>, [[VECTOR_PH]] ], [ [[VEC_IND_NEXT:%.*]], [[MIDDLE_LOOP_J0_CLEANUP4]] ]
; CHECK-NEXT:    br label [[INNERMOST_LOOP3:%.*]]
; CHECK:       innermost.loop3:
; CHECK-NEXT:    [[VEC_PHI:%.*]] = phi <4 x i64> [ zeroinitializer, [[VECTOR_BODY]] ], [ [[TMP5:%.*]], [[INNERMOST_LOOP3]] ]
; CHECK-NEXT:    [[TMP1:%.*]] = add nuw nsw <4 x i64> [[BROADCAST_SPLAT]], [[VEC_PHI]]
; CHECK-NEXT:    [[TMP2:%.*]] = trunc <4 x i64> [[TMP1]] to <4 x i32>
; CHECK-NEXT:    [[TMP3:%.*]] = getelementptr ptr, ptr [[INVARIANT_GEP]], <4 x i64> [[VEC_PHI]]
; CHECK-NEXT:    [[WIDE_MASKED_GATHER:%.*]] = call <4 x ptr> @llvm.masked.gather.v4p0.v4p0(<4 x ptr> align 8 [[TMP3]], <4 x i1> splat (i1 true), <4 x ptr> poison)
; CHECK-NEXT:    [[TMP4:%.*]] = getelementptr inbounds i32, <4 x ptr> [[WIDE_MASKED_GATHER]], <4 x i64> [[VEC_IND]]
; CHECK-NEXT:    call void @llvm.masked.scatter.v4i32.v4p0(<4 x i32> [[TMP2]], <4 x ptr> align 4 [[TMP4]], <4 x i1> splat (i1 true))
; CHECK-NEXT:    [[TMP5]] = add nuw nsw <4 x i64> [[VEC_PHI]], splat (i64 1)
; CHECK-NEXT:    [[TMP6:%.*]] = icmp eq <4 x i64> [[TMP5]], [[BROADCAST_SPLAT3]]
; CHECK-NEXT:    [[TMP7:%.*]] = extractelement <4 x i1> [[TMP6]], i32 0
; CHECK-NEXT:    br i1 [[TMP7]], label [[MIDDLE_LOOP_J0_CLEANUP4]], label [[INNERMOST_LOOP3]]
; CHECK:       vector.latch:
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 4
; CHECK-NEXT:    [[VEC_IND_NEXT]] = add <4 x i64> [[VEC_IND]], splat (i64 4)
; CHECK-NEXT:    [[TMP10:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[TMP10]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP4:![0-9]+]]
; CHECK:       middle.block:
; CHECK-NEXT:    [[CMP_N:%.*]] = icmp eq i64 [[N]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[CMP_N]], label [[INNERMOST_LOOP_J1_LR_PH:%.*]], label [[SCALAR_PH]]
; CHECK:       scalar.ph:
; CHECK-NEXT:    [[BC_RESUME_VAL:%.*]] = phi i64 [ [[N_VEC]], [[MIDDLE_BLOCK]] ], [ 0, [[MIDDLE_LOOP_I]] ]
; CHECK-NEXT:    br label [[MIDDLE_LOOP_J0_PH:%.*]]
; CHECK:       outermost.loop.k.cleanup:
; CHECK-NEXT:    [[K_NEXT]] = add nuw nsw i64 [[K]], 1
; CHECK-NEXT:    [[EXITCOND71_NOT:%.*]] = icmp eq i64 [[K_NEXT]], [[N]]
; CHECK-NEXT:    br i1 [[EXITCOND71_NOT]], label [[RETURN:%.*]], label [[OUTERMOST_LOOP_K]]
; CHECK:       innermost.loop.j1.lr.ph:
; CHECK-NEXT:    [[TMP11:%.*]] = load ptr, ptr [[INVARIANT_GEP]], align 8
; CHECK-NEXT:    br label [[INNERMOST_LOOP_J1:%.*]]
; CHECK:       middle.loop.j0.ph:
; CHECK-NEXT:    [[J0:%.*]] = phi i64 [ [[BC_RESUME_VAL]], [[SCALAR_PH]] ], [ [[J0_NEXT:%.*]], [[MIDDLE_LOOP_J0_CLEANUP:%.*]] ]
; CHECK-NEXT:    br label [[INNERMOST_LOOP:%.*]]
; CHECK:       middle.loop.j0.cleanup:
; CHECK-NEXT:    [[J0_NEXT]] = add nuw nsw i64 [[J0]], 1
; CHECK-NEXT:    [[J0_EXIT_COND_NOT:%.*]] = icmp eq i64 [[J0_NEXT]], [[N]]
; CHECK-NEXT:    br i1 [[J0_EXIT_COND_NOT]], label [[INNERMOST_LOOP_J1_LR_PH]], label [[MIDDLE_LOOP_J0_PH]], !llvm.loop [[LOOP5:![0-9]+]]
; CHECK:       innermost.loop:
; CHECK-NEXT:    [[X:%.*]] = phi i64 [ 0, [[MIDDLE_LOOP_J0_PH]] ], [ [[X_NEXT:%.*]], [[INNERMOST_LOOP]] ]
; CHECK-NEXT:    [[ADD14:%.*]] = add nuw nsw i64 [[ADD]], [[X]]
; CHECK-NEXT:    [[CONV:%.*]] = trunc i64 [[ADD14]] to i32
; CHECK-NEXT:    [[GEP:%.*]] = getelementptr ptr, ptr [[INVARIANT_GEP]], i64 [[X]]
; CHECK-NEXT:    [[TMP12:%.*]] = load ptr, ptr [[GEP]], align 8
; CHECK-NEXT:    [[ARRAYIDX17:%.*]] = getelementptr inbounds i32, ptr [[TMP12]], i64 [[J0]]
; CHECK-NEXT:    store i32 [[CONV]], ptr [[ARRAYIDX17]], align 4
; CHECK-NEXT:    [[X_NEXT]] = add nuw nsw i64 [[X]], 1
; CHECK-NEXT:    [[EXITCOND_NOT:%.*]] = icmp eq i64 [[X_NEXT]], [[N]]
; CHECK-NEXT:    br i1 [[EXITCOND_NOT]], label [[MIDDLE_LOOP_J0_CLEANUP]], label [[INNERMOST_LOOP]]
; CHECK:       middle.loop.i.cleanup:
; CHECK-NEXT:    [[I_NEXT]] = add nuw nsw i64 [[I]], 1
; CHECK-NEXT:    [[EXITCOND70_NOT:%.*]] = icmp eq i64 [[I_NEXT]], [[N]]
; CHECK-NEXT:    br i1 [[EXITCOND70_NOT]], label [[OUTERMOST_LOOP_K_CLEANUP]], label [[MIDDLE_LOOP_I]]
; CHECK:       innermost.loop.j1:
; CHECK-NEXT:    [[J21_064:%.*]] = phi i64 [ [[N]], [[INNERMOST_LOOP_J1_LR_PH]] ], [ [[DEC:%.*]], [[INNERMOST_LOOP_J1]] ]
; CHECK-NEXT:    [[ARRAYIDX28:%.*]] = getelementptr inbounds i32, ptr [[TMP11]], i64 [[J21_064]]
; CHECK-NEXT:    [[TMP13:%.*]] = load i32, ptr [[ARRAYIDX28]], align 4
; CHECK-NEXT:    [[TMP14:%.*]] = and i64 [[J21_064]], 1
; CHECK-NEXT:    [[DOTNOT:%.*]] = icmp eq i64 [[TMP14]], 0
; CHECK-NEXT:    [[CONV30:%.*]] = select i1 [[DOTNOT]], i32 0, i32 [[TMP13]]
; CHECK-NEXT:    store i32 [[CONV30]], ptr [[ARRAYIDX28]], align 4
; CHECK-NEXT:    [[DEC]] = add nsw i64 [[J21_064]], -1
; CHECK-NEXT:    [[CMP23:%.*]] = icmp sgt i64 [[J21_064]], 1
; CHECK-NEXT:    br i1 [[CMP23]], label [[INNERMOST_LOOP_J1]], label [[MIDDLE_LOOP_I_CLEANUP]]
;
entry:
  br label %outermost.loop.k

return:
  ret void

outermost.loop.k:
  %k = phi i64 [ %k.next, %outermost.loop.k.cleanup ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds ptr, ptr %a, i64 %k
  %0 = load ptr, ptr %arrayidx, align 8
  store i32 0, ptr %0, align 4
  %add = add nuw nsw i64 %k, 2
  br label %middle.loop.i

middle.loop.i:
  %i = phi i64 [ 0, %outermost.loop.k ], [ %i.next, %middle.loop.i.cleanup ]
  %invariant.gep = getelementptr ptr, ptr %a, i64 %i
  br label %middle.loop.j0.ph

outermost.loop.k.cleanup:
  %k.next = add nuw nsw i64 %k, 1
  %exitcond71.not = icmp eq i64 %k.next, %n
  br i1 %exitcond71.not, label %return, label %outermost.loop.k

innermost.loop.j1.lr.ph:                                 ; preds = %middle.loop.j0.cleanup
  %1 = load ptr, ptr %invariant.gep, align 8
  br label %innermost.loop.j1

middle.loop.j0.ph:
  %j0 = phi i64 [ 0, %middle.loop.i ], [ %j0.next, %middle.loop.j0.cleanup ]
  br label %innermost.loop

middle.loop.j0.cleanup:
  %j0.next = add nuw nsw i64 %j0, 1
  %j0.exit.cond.not = icmp eq i64 %j0.next, %n
  br i1 %j0.exit.cond.not, label %innermost.loop.j1.lr.ph, label %middle.loop.j0.ph, !llvm.loop !3

innermost.loop:
  %x = phi i64 [ 0, %middle.loop.j0.ph ], [ %x.next, %innermost.loop ]
  %add14 = add nuw nsw i64 %add, %x
  %conv = trunc i64 %add14 to i32
  %gep = getelementptr ptr, ptr %invariant.gep, i64 %x
  %2 = load ptr, ptr %gep, align 8
  %arrayidx17 = getelementptr inbounds i32, ptr %2, i64 %j0
  store i32 %conv, ptr %arrayidx17, align 4
  %x.next = add nuw nsw i64 %x, 1
  %exitcond.not = icmp eq i64 %x.next, %n
  br i1 %exitcond.not, label %middle.loop.j0.cleanup, label %innermost.loop

middle.loop.i.cleanup:
  %i.next = add nuw nsw i64 %i, 1
  %exitcond70.not = icmp eq i64 %i.next, %n
  br i1 %exitcond70.not, label %outermost.loop.k.cleanup, label %middle.loop.i

innermost.loop.j1:
  %j21.064 = phi i64 [ %n, %innermost.loop.j1.lr.ph ], [ %dec, %innermost.loop.j1 ]
  %arrayidx28 = getelementptr inbounds i32, ptr %1, i64 %j21.064
  %3 = load i32, ptr %arrayidx28, align 4
  %4 = and i64 %j21.064, 1
  %.not = icmp eq i64 %4, 0
  %conv30 = select i1 %.not, i32 0, i32 %3
  store i32 %conv30, ptr %arrayidx28, align 4
  %dec = add nsw i64 %j21.064, -1
  %cmp23 = icmp sgt i64 %j21.064, 1
  br i1 %cmp23, label %innermost.loop.j1, label %middle.loop.i.cleanup
}

!3 = distinct !{!3, !4, !5, !6}
!4 = !{!"llvm.loop.vectorize.width", i32 4}
!5 = !{!"llvm.loop.vectorize.scalable.enable", i1 false}
!6 = !{!"llvm.loop.vectorize.enable", i1 true}
