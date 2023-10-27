; RUN: opt < %s -passes=loop-vectorize,dce,instcombine -force-vector-interleave=1 -force-vector-width=4 -prefer-inloop-reductions -S | FileCheck %s

define float @cond_fadd(ptr noalias nocapture readonly %a, ptr noalias nocapture readonly %cond, i64 %N){
; CHECK-LABEL: @cond_fadd(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[MIN_ITERS_CHECK:%.*]] = icmp ult i64 [[N:%.*]], 4
; CHECK-NEXT:    br i1 [[MIN_ITERS_CHECK]], label [[SCALAR_PH:%.*]], label [[VECTOR_PH:%.*]]
; CHECK:       vector.ph:
; CHECK-NEXT:    [[N_VEC:%.*]] = and i64 [[N]], -4
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[PRED_LOAD_CONTINUE6:%.*]] ]
; CHECK-NEXT:    [[VEC_PHI:%.*]] = phi float [ 1.000000e+00, [[VECTOR_PH]] ], [ [[TMP27:%.*]], [[PRED_LOAD_CONTINUE6]] ]
; CHECK-NEXT:    [[TMP0:%.*]] = getelementptr inbounds float, ptr [[COND:%.*]], i64 [[INDEX]]
; CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <4 x float>, ptr [[TMP0]], align 4
; CHECK-NEXT:    [[TMP2:%.*]] = fcmp une <4 x float> [[WIDE_LOAD]], <float 5.000000e+00, float 5.000000e+00, float 5.000000e+00, float 5.000000e+00>
; CHECK-NEXT:    [[TMP3:%.*]] = extractelement <4 x i1> [[TMP2]], i64 0
; CHECK-NEXT:    br i1 [[TMP3]], label [[PRED_LOAD_IF:%.*]], label [[PRED_LOAD_CONTINUE:%.*]]
; CHECK:       pred.load.if:
; CHECK-NEXT:    [[TMP4:%.*]] = getelementptr inbounds float, ptr [[A:%.*]], i64 [[INDEX]]
; CHECK-NEXT:    [[TMP5:%.*]] = load float, ptr [[TMP4]], align 4
; CHECK-NEXT:    [[TMP6:%.*]] = insertelement <4 x float> poison, float [[TMP5]], i64 0
; CHECK-NEXT:    br label [[PRED_LOAD_CONTINUE]]
; CHECK:       pred.load.continue:
; CHECK-NEXT:    [[TMP7:%.*]] = phi <4 x float> [ poison, [[VECTOR_BODY]] ], [ [[TMP6]], [[PRED_LOAD_IF]] ]
; CHECK-NEXT:    [[TMP8:%.*]] = extractelement <4 x i1> [[TMP2]], i64 1
; CHECK-NEXT:    br i1 [[TMP8]], label [[PRED_LOAD_IF1:%.*]], label [[PRED_LOAD_CONTINUE2:%.*]]
; CHECK:       pred.load.if1:
; CHECK-NEXT:    [[TMP9:%.*]] = or i64 [[INDEX]], 1
; CHECK-NEXT:    [[TMP10:%.*]] = getelementptr inbounds float, ptr [[A]], i64 [[TMP9]]
; CHECK-NEXT:    [[TMP11:%.*]] = load float, ptr [[TMP10]], align 4
; CHECK-NEXT:    [[TMP12:%.*]] = insertelement <4 x float> [[TMP7]], float [[TMP11]], i64 1
; CHECK-NEXT:    br label [[PRED_LOAD_CONTINUE2]]
; CHECK:       pred.load.continue2:
; CHECK-NEXT:    [[TMP13:%.*]] = phi <4 x float> [ [[TMP7]], [[PRED_LOAD_CONTINUE]] ], [ [[TMP12]], [[PRED_LOAD_IF1]] ]
; CHECK-NEXT:    [[TMP14:%.*]] = extractelement <4 x i1> [[TMP2]], i64 2
; CHECK-NEXT:    br i1 [[TMP14]], label [[PRED_LOAD_IF3:%.*]], label [[PRED_LOAD_CONTINUE4:%.*]]
; CHECK:       pred.load.if3:
; CHECK-NEXT:    [[TMP15:%.*]] = or i64 [[INDEX]], 2
; CHECK-NEXT:    [[TMP16:%.*]] = getelementptr inbounds float, ptr [[A]], i64 [[TMP15]]
; CHECK-NEXT:    [[TMP17:%.*]] = load float, ptr [[TMP16]], align 4
; CHECK-NEXT:    [[TMP18:%.*]] = insertelement <4 x float> [[TMP13]], float [[TMP17]], i64 2
; CHECK-NEXT:    br label [[PRED_LOAD_CONTINUE4]]
; CHECK:       pred.load.continue4:
; CHECK-NEXT:    [[TMP19:%.*]] = phi <4 x float> [ [[TMP13]], [[PRED_LOAD_CONTINUE2]] ], [ [[TMP18]], [[PRED_LOAD_IF3]] ]
; CHECK-NEXT:    [[TMP20:%.*]] = extractelement <4 x i1> [[TMP2]], i64 3
; CHECK-NEXT:    br i1 [[TMP20]], label [[PRED_LOAD_IF5:%.*]], label [[PRED_LOAD_CONTINUE6]]
; CHECK:       pred.load.if5:
; CHECK-NEXT:    [[TMP21:%.*]] = or i64 [[INDEX]], 3
; CHECK-NEXT:    [[TMP22:%.*]] = getelementptr inbounds float, ptr [[A]], i64 [[TMP21]]
; CHECK-NEXT:    [[TMP23:%.*]] = load float, ptr [[TMP22]], align 4
; CHECK-NEXT:    [[TMP24:%.*]] = insertelement <4 x float> [[TMP19]], float [[TMP23]], i64 3
; CHECK-NEXT:    br label [[PRED_LOAD_CONTINUE6]]
; CHECK:       pred.load.continue6:
; CHECK-NEXT:    [[TMP25:%.*]] = phi <4 x float> [ [[TMP19]], [[PRED_LOAD_CONTINUE4]] ], [ [[TMP24]], [[PRED_LOAD_IF5]] ]
; CHECK-NEXT:    [[TMP26:%.*]] = select fast <4 x i1> [[TMP2]], <4 x float> [[TMP25]], <4 x float> zeroinitializer
; CHECK-NEXT:    [[TMP27]] = call fast float @llvm.vector.reduce.fadd.v4f32(float [[VEC_PHI]], <4 x float> [[TMP26]])
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 4
; CHECK-NEXT:    [[TMP28:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[TMP28]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP0:![0-9]+]]
; CHECK:       middle.block:
; CHECK-NEXT:    [[CMP_N:%.*]] = icmp eq i64 [[N_VEC]], [[N]]
; CHECK-NEXT:    br i1 [[CMP_N]], label [[FOR_END:%.*]], label [[SCALAR_PH]]
; CHECK:       scalar.ph:
; CHECK-NEXT:    [[BC_RESUME_VAL:%.*]] = phi i64 [ [[N_VEC]], [[MIDDLE_BLOCK]] ], [ 0, [[ENTRY:%.*]] ]
; CHECK-NEXT:    [[BC_MERGE_RDX:%.*]] = phi float [ [[TMP27]], [[MIDDLE_BLOCK]] ], [ 1.000000e+00, [[ENTRY]] ]
; CHECK-NEXT:    br label [[FOR_BODY:%.*]]
; CHECK:       for.body:
; CHECK-NEXT:    [[IV:%.*]] = phi i64 [ [[BC_RESUME_VAL]], [[SCALAR_PH]] ], [ [[IV_NEXT:%.*]], [[FOR_INC:%.*]] ]
; CHECK-NEXT:    [[RDX:%.*]] = phi float [ [[BC_MERGE_RDX]], [[SCALAR_PH]] ], [ [[RES:%.*]], [[FOR_INC]] ]
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds float, ptr [[COND]], i64 [[IV]]
; CHECK-NEXT:    [[TMP29:%.*]] = load float, ptr [[ARRAYIDX]], align 4
; CHECK-NEXT:    [[TOBOOL:%.*]] = fcmp une float [[TMP29]], 5.000000e+00
; CHECK-NEXT:    br i1 [[TOBOOL]], label [[IF_THEN:%.*]], label [[FOR_INC]]
; CHECK:       if.then:
; CHECK-NEXT:    [[ARRAYIDX2:%.*]] = getelementptr inbounds float, ptr [[A]], i64 [[IV]]
; CHECK-NEXT:    [[TMP30:%.*]] = load float, ptr [[ARRAYIDX2]], align 4
; CHECK-NEXT:    [[FADD:%.*]] = fadd fast float [[RDX]], [[TMP30]]
; CHECK-NEXT:    br label [[FOR_INC]]
; CHECK:       for.inc:
; CHECK-NEXT:    [[RES]] = phi float [ [[RDX]], [[FOR_BODY]] ], [ [[FADD]], [[IF_THEN]] ]
; CHECK-NEXT:    [[IV_NEXT]] = add i64 [[IV]], 1
; CHECK-NEXT:    [[EXITCOND_NOT:%.*]] = icmp eq i64 [[IV_NEXT]], [[N]]
; CHECK-NEXT:    br i1 [[EXITCOND_NOT]], label [[FOR_END]], label [[FOR_BODY]], !llvm.loop [[LOOP2:![0-9]+]]
; CHECK:       for.end:
; CHECK-NEXT:    [[RES_LCSSA:%.*]] = phi float [ [[RES]], [[FOR_INC]] ], [ [[TMP27]], [[MIDDLE_BLOCK]] ]
; CHECK-NEXT:    ret float [[RES_LCSSA]]
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %rdx = phi float [ 1.000000e+00, %entry ], [ %res, %for.inc ]
  %arrayidx = getelementptr inbounds float, ptr %cond, i64 %iv
  %0 = load float, ptr %arrayidx
  %tobool = fcmp une float %0, 5.000000e+00
  br i1 %tobool, label %if.then, label %for.inc

if.then:
  %arrayidx2 = getelementptr inbounds float, ptr %a, i64 %iv
  %1 = load float, ptr %arrayidx2
  %fadd = fadd fast float %rdx, %1
  br label %for.inc

for.inc:
  %res = phi float [ %rdx, %for.body ], [ %fadd, %if.then ]
  %iv.next = add i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:
  ret float %res
}

define float @cond_cmp_sel(ptr noalias %a, ptr noalias %cond, i64 %N) {
; CHECK-LABEL: @cond_cmp_sel(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[MIN_ITERS_CHECK:%.*]] = icmp ult i64 [[N:%.*]], 4
; CHECK-NEXT:    br i1 [[MIN_ITERS_CHECK]], label [[SCALAR_PH:%.*]], label [[VECTOR_PH:%.*]]
; CHECK:       vector.ph:
; CHECK-NEXT:    [[N_VEC:%.*]] = and i64 [[N]], -4
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[PRED_LOAD_CONTINUE6:%.*]] ]
; CHECK-NEXT:    [[VEC_PHI:%.*]] = phi float [ 1.000000e+00, [[VECTOR_PH]] ], [ [[TMP28:%.*]], [[PRED_LOAD_CONTINUE6]] ]
; CHECK-NEXT:    [[TMP0:%.*]] = getelementptr inbounds float, ptr [[COND:%.*]], i64 [[INDEX]]
; CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <4 x float>, ptr [[TMP0]], align 4
; CHECK-NEXT:    [[TMP2:%.*]] = fcmp une <4 x float> [[WIDE_LOAD]], <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
; CHECK-NEXT:    [[TMP3:%.*]] = extractelement <4 x i1> [[TMP2]], i64 0
; CHECK-NEXT:    br i1 [[TMP3]], label [[PRED_LOAD_IF:%.*]], label [[PRED_LOAD_CONTINUE:%.*]]
; CHECK:       pred.load.if:
; CHECK-NEXT:    [[TMP4:%.*]] = getelementptr inbounds float, ptr [[A:%.*]], i64 [[INDEX]]
; CHECK-NEXT:    [[TMP5:%.*]] = load float, ptr [[TMP4]], align 4
; CHECK-NEXT:    [[TMP6:%.*]] = insertelement <4 x float> poison, float [[TMP5]], i64 0
; CHECK-NEXT:    br label [[PRED_LOAD_CONTINUE]]
; CHECK:       pred.load.continue:
; CHECK-NEXT:    [[TMP7:%.*]] = phi <4 x float> [ poison, [[VECTOR_BODY]] ], [ [[TMP6]], [[PRED_LOAD_IF]] ]
; CHECK-NEXT:    [[TMP8:%.*]] = extractelement <4 x i1> [[TMP2]], i64 1
; CHECK-NEXT:    br i1 [[TMP8]], label [[PRED_LOAD_IF1:%.*]], label [[PRED_LOAD_CONTINUE2:%.*]]
; CHECK:       pred.load.if1:
; CHECK-NEXT:    [[TMP9:%.*]] = or i64 [[INDEX]], 1
; CHECK-NEXT:    [[TMP10:%.*]] = getelementptr inbounds float, ptr [[A]], i64 [[TMP9]]
; CHECK-NEXT:    [[TMP11:%.*]] = load float, ptr [[TMP10]], align 4
; CHECK-NEXT:    [[TMP12:%.*]] = insertelement <4 x float> [[TMP7]], float [[TMP11]], i64 1
; CHECK-NEXT:    br label [[PRED_LOAD_CONTINUE2]]
; CHECK:       pred.load.continue2:
; CHECK-NEXT:    [[TMP13:%.*]] = phi <4 x float> [ [[TMP7]], [[PRED_LOAD_CONTINUE]] ], [ [[TMP12]], [[PRED_LOAD_IF1]] ]
; CHECK-NEXT:    [[TMP14:%.*]] = extractelement <4 x i1> [[TMP2]], i64 2
; CHECK-NEXT:    br i1 [[TMP14]], label [[PRED_LOAD_IF3:%.*]], label [[PRED_LOAD_CONTINUE4:%.*]]
; CHECK:       pred.load.if3:
; CHECK-NEXT:    [[TMP15:%.*]] = or i64 [[INDEX]], 2
; CHECK-NEXT:    [[TMP16:%.*]] = getelementptr inbounds float, ptr [[A]], i64 [[TMP15]]
; CHECK-NEXT:    [[TMP17:%.*]] = load float, ptr [[TMP16]], align 4
; CHECK-NEXT:    [[TMP18:%.*]] = insertelement <4 x float> [[TMP13]], float [[TMP17]], i64 2
; CHECK-NEXT:    br label [[PRED_LOAD_CONTINUE4]]
; CHECK:       pred.load.continue4:
; CHECK-NEXT:    [[TMP19:%.*]] = phi <4 x float> [ [[TMP13]], [[PRED_LOAD_CONTINUE2]] ], [ [[TMP18]], [[PRED_LOAD_IF3]] ]
; CHECK-NEXT:    [[TMP20:%.*]] = extractelement <4 x i1> [[TMP2]], i64 3
; CHECK-NEXT:    br i1 [[TMP20]], label [[PRED_LOAD_IF5:%.*]], label [[PRED_LOAD_CONTINUE6]]
; CHECK:       pred.load.if5:
; CHECK-NEXT:    [[TMP21:%.*]] = or i64 [[INDEX]], 3
; CHECK-NEXT:    [[TMP22:%.*]] = getelementptr inbounds float, ptr [[A]], i64 [[TMP21]]
; CHECK-NEXT:    [[TMP23:%.*]] = load float, ptr [[TMP22]], align 4
; CHECK-NEXT:    [[TMP24:%.*]] = insertelement <4 x float> [[TMP19]], float [[TMP23]], i64 3
; CHECK-NEXT:    br label [[PRED_LOAD_CONTINUE6]]
; CHECK:       pred.load.continue6:
; CHECK-NEXT:    [[TMP25:%.*]] = phi <4 x float> [ [[TMP19]], [[PRED_LOAD_CONTINUE4]] ], [ [[TMP24]], [[PRED_LOAD_IF5]] ]
; CHECK-NEXT:    [[TMP26:%.*]] = select fast <4 x i1> [[TMP2]], <4 x float> [[TMP25]], <4 x float> <float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000>
; CHECK-NEXT:    [[TMP27:%.*]] = call fast float @llvm.vector.reduce.fmin.v4f32(<4 x float> [[TMP26]])
; CHECK-NEXT:    [[TMP28]] = call fast float @llvm.minnum.f32(float [[TMP27]], float [[VEC_PHI]])
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 4
; CHECK-NEXT:    [[TMP29:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[TMP29]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP4:![0-9]+]]
; CHECK:       middle.block:
; CHECK-NEXT:    [[CMP_N:%.*]] = icmp eq i64 [[N_VEC]], [[N]]
; CHECK-NEXT:    br i1 [[CMP_N]], label [[FOR_END:%.*]], label [[SCALAR_PH]]
; CHECK:       scalar.ph:
; CHECK-NEXT:    [[BC_RESUME_VAL:%.*]] = phi i64 [ [[N_VEC]], [[MIDDLE_BLOCK]] ], [ 0, [[ENTRY:%.*]] ]
; CHECK-NEXT:    [[BC_MERGE_RDX:%.*]] = phi float [ [[TMP28]], [[MIDDLE_BLOCK]] ], [ 1.000000e+00, [[ENTRY]] ]
; CHECK-NEXT:    br label [[FOR_BODY:%.*]]
; CHECK:       for.body:
; CHECK-NEXT:    [[IV:%.*]] = phi i64 [ [[BC_RESUME_VAL]], [[SCALAR_PH]] ], [ [[IV_NEXT:%.*]], [[FOR_INC:%.*]] ]
; CHECK-NEXT:    [[RDX:%.*]] = phi float [ [[BC_MERGE_RDX]], [[SCALAR_PH]] ], [ [[RES:%.*]], [[FOR_INC]] ]
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds float, ptr [[COND]], i64 [[IV]]
; CHECK-NEXT:    [[TMP30:%.*]] = load float, ptr [[ARRAYIDX]], align 4
; CHECK-NEXT:    [[TOBOOL:%.*]] = fcmp une float [[TMP30]], 3.000000e+00
; CHECK-NEXT:    br i1 [[TOBOOL]], label [[IF_THEN:%.*]], label [[FOR_INC]]
; CHECK:       if.then:
; CHECK-NEXT:    [[ARRAYIDX2:%.*]] = getelementptr inbounds float, ptr [[A]], i64 [[IV]]
; CHECK-NEXT:    [[TMP31:%.*]] = load float, ptr [[ARRAYIDX2]], align 4
; CHECK-NEXT:    [[TMP32:%.*]] = call fast float @llvm.minnum.f32(float [[RDX]], float [[TMP31]])
; CHECK-NEXT:    br label [[FOR_INC]]
; CHECK:       for.inc:
; CHECK-NEXT:    [[RES]] = phi float [ [[RDX]], [[FOR_BODY]] ], [ [[TMP32]], [[IF_THEN]] ]
; CHECK-NEXT:    [[IV_NEXT]] = add i64 [[IV]], 1
; CHECK-NEXT:    [[EXITCOND_NOT:%.*]] = icmp eq i64 [[IV_NEXT]], [[N]]
; CHECK-NEXT:    br i1 [[EXITCOND_NOT]], label [[FOR_END]], label [[FOR_BODY]], !llvm.loop [[LOOP5:![0-9]+]]
; CHECK:       for.end:
; CHECK-NEXT:    [[RES_LCSSA:%.*]] = phi float [ [[RES]], [[FOR_INC]] ], [ [[TMP28]], [[MIDDLE_BLOCK]] ]
; CHECK-NEXT:    ret float [[RES_LCSSA]]
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %rdx = phi float [ 1.000000e+00, %entry ], [ %res, %for.inc ]
  %arrayidx = getelementptr inbounds float, ptr %cond, i64 %iv
  %0 = load float, ptr %arrayidx
  %tobool = fcmp une float %0, 3.000000e+00
  br i1 %tobool, label %if.then, label %for.inc

if.then:
  %arrayidx2 = getelementptr inbounds float, ptr %a, i64 %iv
  %1 = load float, ptr %arrayidx2
  %fcmp = fcmp fast olt float %rdx, %1
  %fsel = select fast i1 %fcmp, float %rdx, float %1
  br label %for.inc

for.inc:
  %res = phi float [ %rdx, %for.body ], [ %fsel, %if.then ]
  %iv.next = add i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:
  ret float %res
}

define i32 @conditional_and(ptr noalias %A, ptr noalias %B, i32 %cond, i64 noundef %N) #0 {
; CHECK-LABEL: @conditional_and(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[MIN_ITERS_CHECK:%.*]] = icmp ult i64 [[N:%.*]], 4
; CHECK-NEXT:    br i1 [[MIN_ITERS_CHECK]], label [[SCALAR_PH:%.*]], label [[VECTOR_PH:%.*]]
; CHECK:       vector.ph:
; CHECK-NEXT:    [[N_VEC:%.*]] = and i64 [[N]], -4
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <4 x i32> poison, i32 [[COND:%.*]], i64 0
; CHECK-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <4 x i32> [[BROADCAST_SPLATINSERT]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[PRED_LOAD_CONTINUE6:%.*]] ]
; CHECK-NEXT:    [[VEC_PHI:%.*]] = phi i32 [ 7, [[VECTOR_PH]] ], [ [[TMP28:%.*]], [[PRED_LOAD_CONTINUE6]] ]
; CHECK-NEXT:    [[TMP0:%.*]] = getelementptr inbounds i32, ptr [[A:%.*]], i64 [[INDEX]]
; CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <4 x i32>, ptr [[TMP0]], align 4
; CHECK-NEXT:    [[TMP2:%.*]] = icmp eq <4 x i32> [[WIDE_LOAD]], [[BROADCAST_SPLAT]]
; CHECK-NEXT:    [[TMP3:%.*]] = extractelement <4 x i1> [[TMP2]], i64 0
; CHECK-NEXT:    br i1 [[TMP3]], label [[PRED_LOAD_IF:%.*]], label [[PRED_LOAD_CONTINUE:%.*]]
; CHECK:       pred.load.if:
; CHECK-NEXT:    [[TMP4:%.*]] = getelementptr inbounds i32, ptr [[B:%.*]], i64 [[INDEX]]
; CHECK-NEXT:    [[TMP5:%.*]] = load i32, ptr [[TMP4]], align 4
; CHECK-NEXT:    [[TMP6:%.*]] = insertelement <4 x i32> poison, i32 [[TMP5]], i64 0
; CHECK-NEXT:    br label [[PRED_LOAD_CONTINUE]]
; CHECK:       pred.load.continue:
; CHECK-NEXT:    [[TMP7:%.*]] = phi <4 x i32> [ poison, [[VECTOR_BODY]] ], [ [[TMP6]], [[PRED_LOAD_IF]] ]
; CHECK-NEXT:    [[TMP8:%.*]] = extractelement <4 x i1> [[TMP2]], i64 1
; CHECK-NEXT:    br i1 [[TMP8]], label [[PRED_LOAD_IF1:%.*]], label [[PRED_LOAD_CONTINUE2:%.*]]
; CHECK:       pred.load.if1:
; CHECK-NEXT:    [[TMP9:%.*]] = or i64 [[INDEX]], 1
; CHECK-NEXT:    [[TMP10:%.*]] = getelementptr inbounds i32, ptr [[B]], i64 [[TMP9]]
; CHECK-NEXT:    [[TMP11:%.*]] = load i32, ptr [[TMP10]], align 4
; CHECK-NEXT:    [[TMP12:%.*]] = insertelement <4 x i32> [[TMP7]], i32 [[TMP11]], i64 1
; CHECK-NEXT:    br label [[PRED_LOAD_CONTINUE2]]
; CHECK:       pred.load.continue2:
; CHECK-NEXT:    [[TMP13:%.*]] = phi <4 x i32> [ [[TMP7]], [[PRED_LOAD_CONTINUE]] ], [ [[TMP12]], [[PRED_LOAD_IF1]] ]
; CHECK-NEXT:    [[TMP14:%.*]] = extractelement <4 x i1> [[TMP2]], i64 2
; CHECK-NEXT:    br i1 [[TMP14]], label [[PRED_LOAD_IF3:%.*]], label [[PRED_LOAD_CONTINUE4:%.*]]
; CHECK:       pred.load.if3:
; CHECK-NEXT:    [[TMP15:%.*]] = or i64 [[INDEX]], 2
; CHECK-NEXT:    [[TMP16:%.*]] = getelementptr inbounds i32, ptr [[B]], i64 [[TMP15]]
; CHECK-NEXT:    [[TMP17:%.*]] = load i32, ptr [[TMP16]], align 4
; CHECK-NEXT:    [[TMP18:%.*]] = insertelement <4 x i32> [[TMP13]], i32 [[TMP17]], i64 2
; CHECK-NEXT:    br label [[PRED_LOAD_CONTINUE4]]
; CHECK:       pred.load.continue4:
; CHECK-NEXT:    [[TMP19:%.*]] = phi <4 x i32> [ [[TMP13]], [[PRED_LOAD_CONTINUE2]] ], [ [[TMP18]], [[PRED_LOAD_IF3]] ]
; CHECK-NEXT:    [[TMP20:%.*]] = extractelement <4 x i1> [[TMP2]], i64 3
; CHECK-NEXT:    br i1 [[TMP20]], label [[PRED_LOAD_IF5:%.*]], label [[PRED_LOAD_CONTINUE6]]
; CHECK:       pred.load.if5:
; CHECK-NEXT:    [[TMP21:%.*]] = or i64 [[INDEX]], 3
; CHECK-NEXT:    [[TMP22:%.*]] = getelementptr inbounds i32, ptr [[B]], i64 [[TMP21]]
; CHECK-NEXT:    [[TMP23:%.*]] = load i32, ptr [[TMP22]], align 4
; CHECK-NEXT:    [[TMP24:%.*]] = insertelement <4 x i32> [[TMP19]], i32 [[TMP23]], i64 3
; CHECK-NEXT:    br label [[PRED_LOAD_CONTINUE6]]
; CHECK:       pred.load.continue6:
; CHECK-NEXT:    [[TMP25:%.*]] = phi <4 x i32> [ [[TMP19]], [[PRED_LOAD_CONTINUE4]] ], [ [[TMP24]], [[PRED_LOAD_IF5]] ]
; CHECK-NEXT:    [[TMP26:%.*]] = select <4 x i1> [[TMP2]], <4 x i32> [[TMP25]], <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>
; CHECK-NEXT:    [[TMP27:%.*]] = call i32 @llvm.vector.reduce.and.v4i32(<4 x i32> [[TMP26]])
; CHECK-NEXT:    [[TMP28]] = and i32 [[TMP27]], [[VEC_PHI]]
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 4
; CHECK-NEXT:    [[TMP29:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[TMP29]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP6:![0-9]+]]
; CHECK:       middle.block:
; CHECK-NEXT:    [[CMP_N:%.*]] = icmp eq i64 [[N_VEC]], [[N]]
; CHECK-NEXT:    br i1 [[CMP_N]], label [[FOR_END:%.*]], label [[SCALAR_PH]]
; CHECK:       scalar.ph:
; CHECK-NEXT:    [[BC_RESUME_VAL:%.*]] = phi i64 [ [[N_VEC]], [[MIDDLE_BLOCK]] ], [ 0, [[ENTRY:%.*]] ]
; CHECK-NEXT:    [[BC_MERGE_RDX:%.*]] = phi i32 [ [[TMP28]], [[MIDDLE_BLOCK]] ], [ 7, [[ENTRY]] ]
; CHECK-NEXT:    br label [[FOR_BODY:%.*]]
; CHECK:       for.body:
; CHECK-NEXT:    [[IV:%.*]] = phi i64 [ [[BC_RESUME_VAL]], [[SCALAR_PH]] ], [ [[IV_NEXT:%.*]], [[FOR_INC:%.*]] ]
; CHECK-NEXT:    [[RDX:%.*]] = phi i32 [ [[BC_MERGE_RDX]], [[SCALAR_PH]] ], [ [[RES:%.*]], [[FOR_INC]] ]
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds i32, ptr [[A]], i64 [[IV]]
; CHECK-NEXT:    [[TMP30:%.*]] = load i32, ptr [[ARRAYIDX]], align 4
; CHECK-NEXT:    [[TOBOOL:%.*]] = icmp eq i32 [[TMP30]], [[COND]]
; CHECK-NEXT:    br i1 [[TOBOOL]], label [[IF_THEN:%.*]], label [[FOR_INC]]
; CHECK:       if.then:
; CHECK-NEXT:    [[ARRAYIDX2:%.*]] = getelementptr inbounds i32, ptr [[B]], i64 [[IV]]
; CHECK-NEXT:    [[TMP31:%.*]] = load i32, ptr [[ARRAYIDX2]], align 4
; CHECK-NEXT:    [[AND:%.*]] = and i32 [[TMP31]], [[RDX]]
; CHECK-NEXT:    br label [[FOR_INC]]
; CHECK:       for.inc:
; CHECK-NEXT:    [[RES]] = phi i32 [ [[AND]], [[IF_THEN]] ], [ [[RDX]], [[FOR_BODY]] ]
; CHECK-NEXT:    [[IV_NEXT]] = add nuw nsw i64 [[IV]], 1
; CHECK-NEXT:    [[EXITCOND_NOT:%.*]] = icmp eq i64 [[IV_NEXT]], [[N]]
; CHECK-NEXT:    br i1 [[EXITCOND_NOT]], label [[FOR_END]], label [[FOR_BODY]], !llvm.loop [[LOOP7:![0-9]+]]
; CHECK:       for.end:
; CHECK-NEXT:    [[RES_LCSSA:%.*]] = phi i32 [ [[RES]], [[FOR_INC]] ], [ [[TMP28]], [[MIDDLE_BLOCK]] ]
; CHECK-NEXT:    ret i32 [[RES_LCSSA]]
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %rdx = phi i32 [ 7, %entry ], [ %res, %for.inc ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %iv
  %0 = load i32, ptr %arrayidx
  %tobool = icmp eq i32 %0, %cond
  br i1 %tobool, label %if.then, label %for.inc

if.then:
  %arrayidx2 = getelementptr inbounds i32, ptr %B, i64 %iv
  %1 = load i32, ptr %arrayidx2
  %and = and i32 %1, %rdx
  br label %for.inc

for.inc:
  %res = phi i32 [ %and, %if.then ], [ %rdx, %for.body ]
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:
  ret i32 %res
}

define i32 @simple_chained_rdx(ptr noalias %a, ptr noalias %b, ptr noalias %cond, i64 noundef %N) {
; CHECK-LABEL: @simple_chained_rdx(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[MIN_ITERS_CHECK:%.*]] = icmp ult i64 [[N:%.*]], 4
; CHECK-NEXT:    br i1 [[MIN_ITERS_CHECK]], label [[SCALAR_PH:%.*]], label [[VECTOR_PH:%.*]]
; CHECK:       vector.ph:
; CHECK-NEXT:    [[N_VEC:%.*]] = and i64 [[N]], -4
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[PRED_LOAD_CONTINUE6:%.*]] ]
; CHECK-NEXT:    [[VEC_PHI:%.*]] = phi i32 [ 5, [[VECTOR_PH]] ], [ [[TMP46:%.*]], [[PRED_LOAD_CONTINUE6]] ]
; CHECK-NEXT:    [[TMP0:%.*]] = getelementptr inbounds i32, ptr [[COND:%.*]], i64 [[INDEX]]
; CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <4 x i32>, ptr [[TMP0]], align 4
; CHECK-NEXT:    [[TMP1:%.*]] = icmp ne <4 x i32> [[WIDE_LOAD]], zeroinitializer
; CHECK-NEXT:    [[TMP2:%.*]] = extractelement <4 x i1> [[TMP1]], i64 0
; CHECK-NEXT:    br i1 [[TMP2]], label [[PRED_LOAD_IF:%.*]], label [[PRED_LOAD_CONTINUE:%.*]]
; CHECK:       pred.load.if:
; CHECK-NEXT:    [[TMP3:%.*]] = getelementptr inbounds i32, ptr [[A:%.*]], i64 [[INDEX]]
; CHECK-NEXT:    [[TMP4:%.*]] = load i32, ptr [[TMP3]], align 4
; CHECK-NEXT:    [[TMP5:%.*]] = insertelement <4 x i32> poison, i32 [[TMP4]], i64 0
; CHECK-NEXT:    [[TMP6:%.*]] = getelementptr inbounds i32, ptr [[B:%.*]], i64 [[INDEX]]
; CHECK-NEXT:    [[TMP7:%.*]] = load i32, ptr [[TMP6]], align 4
; CHECK-NEXT:    [[TMP8:%.*]] = insertelement <4 x i32> poison, i32 [[TMP7]], i64 0
; CHECK-NEXT:    br label [[PRED_LOAD_CONTINUE]]
; CHECK:       pred.load.continue:
; CHECK-NEXT:    [[TMP9:%.*]] = phi <4 x i32> [ poison, [[VECTOR_BODY]] ], [ [[TMP5]], [[PRED_LOAD_IF]] ]
; CHECK-NEXT:    [[TMP10:%.*]] = phi <4 x i32> [ poison, [[VECTOR_BODY]] ], [ [[TMP8]], [[PRED_LOAD_IF]] ]
; CHECK-NEXT:    [[TMP11:%.*]] = extractelement <4 x i1> [[TMP1]], i64 1
; CHECK-NEXT:    br i1 [[TMP11]], label [[PRED_LOAD_IF1:%.*]], label [[PRED_LOAD_CONTINUE2:%.*]]
; CHECK:       pred.load.if1:
; CHECK-NEXT:    [[TMP12:%.*]] = or i64 [[INDEX]], 1
; CHECK-NEXT:    [[TMP13:%.*]] = getelementptr inbounds i32, ptr [[A]], i64 [[TMP12]]
; CHECK-NEXT:    [[TMP14:%.*]] = load i32, ptr [[TMP13]], align 4
; CHECK-NEXT:    [[TMP15:%.*]] = insertelement <4 x i32> [[TMP9]], i32 [[TMP14]], i64 1
; CHECK-NEXT:    [[TMP16:%.*]] = getelementptr inbounds i32, ptr [[B]], i64 [[TMP12]]
; CHECK-NEXT:    [[TMP17:%.*]] = load i32, ptr [[TMP16]], align 4
; CHECK-NEXT:    [[TMP18:%.*]] = insertelement <4 x i32> [[TMP10]], i32 [[TMP17]], i64 1
; CHECK-NEXT:    br label [[PRED_LOAD_CONTINUE2]]
; CHECK:       pred.load.continue2:
; CHECK-NEXT:    [[TMP19:%.*]] = phi <4 x i32> [ [[TMP9]], [[PRED_LOAD_CONTINUE]] ], [ [[TMP15]], [[PRED_LOAD_IF1]] ]
; CHECK-NEXT:    [[TMP20:%.*]] = phi <4 x i32> [ [[TMP10]], [[PRED_LOAD_CONTINUE]] ], [ [[TMP18]], [[PRED_LOAD_IF1]] ]
; CHECK-NEXT:    [[TMP21:%.*]] = extractelement <4 x i1> [[TMP1]], i64 2
; CHECK-NEXT:    br i1 [[TMP21]], label [[PRED_LOAD_IF3:%.*]], label [[PRED_LOAD_CONTINUE4:%.*]]
; CHECK:       pred.load.if3:
; CHECK-NEXT:    [[TMP22:%.*]] = or i64 [[INDEX]], 2
; CHECK-NEXT:    [[TMP23:%.*]] = getelementptr inbounds i32, ptr [[A]], i64 [[TMP22]]
; CHECK-NEXT:    [[TMP24:%.*]] = load i32, ptr [[TMP23]], align 4
; CHECK-NEXT:    [[TMP25:%.*]] = insertelement <4 x i32> [[TMP19]], i32 [[TMP24]], i64 2
; CHECK-NEXT:    [[TMP26:%.*]] = getelementptr inbounds i32, ptr [[B]], i64 [[TMP22]]
; CHECK-NEXT:    [[TMP27:%.*]] = load i32, ptr [[TMP26]], align 4
; CHECK-NEXT:    [[TMP28:%.*]] = insertelement <4 x i32> [[TMP20]], i32 [[TMP27]], i64 2
; CHECK-NEXT:    br label [[PRED_LOAD_CONTINUE4]]
; CHECK:       pred.load.continue4:
; CHECK-NEXT:    [[TMP29:%.*]] = phi <4 x i32> [ [[TMP19]], [[PRED_LOAD_CONTINUE2]] ], [ [[TMP25]], [[PRED_LOAD_IF3]] ]
; CHECK-NEXT:    [[TMP30:%.*]] = phi <4 x i32> [ [[TMP20]], [[PRED_LOAD_CONTINUE2]] ], [ [[TMP28]], [[PRED_LOAD_IF3]] ]
; CHECK-NEXT:    [[TMP31:%.*]] = extractelement <4 x i1> [[TMP1]], i64 3
; CHECK-NEXT:    br i1 [[TMP31]], label [[PRED_LOAD_IF5:%.*]], label [[PRED_LOAD_CONTINUE6]]
; CHECK:       pred.load.if5:
; CHECK-NEXT:    [[TMP32:%.*]] = or i64 [[INDEX]], 3
; CHECK-NEXT:    [[TMP33:%.*]] = getelementptr inbounds i32, ptr [[A]], i64 [[TMP32]]
; CHECK-NEXT:    [[TMP34:%.*]] = load i32, ptr [[TMP33]], align 4
; CHECK-NEXT:    [[TMP35:%.*]] = insertelement <4 x i32> [[TMP29]], i32 [[TMP34]], i64 3
; CHECK-NEXT:    [[TMP36:%.*]] = getelementptr inbounds i32, ptr [[B]], i64 [[TMP32]]
; CHECK-NEXT:    [[TMP37:%.*]] = load i32, ptr [[TMP36]], align 4
; CHECK-NEXT:    [[TMP38:%.*]] = insertelement <4 x i32> [[TMP30]], i32 [[TMP37]], i64 3
; CHECK-NEXT:    br label [[PRED_LOAD_CONTINUE6]]
; CHECK:       pred.load.continue6:
; CHECK-NEXT:    [[TMP39:%.*]] = phi <4 x i32> [ [[TMP29]], [[PRED_LOAD_CONTINUE4]] ], [ [[TMP35]], [[PRED_LOAD_IF5]] ]
; CHECK-NEXT:    [[TMP40:%.*]] = phi <4 x i32> [ [[TMP30]], [[PRED_LOAD_CONTINUE4]] ], [ [[TMP38]], [[PRED_LOAD_IF5]] ]
; CHECK-NEXT:    [[TMP41:%.*]] = select <4 x i1> [[TMP1]], <4 x i32> [[TMP39]], <4 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP42:%.*]] = call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> [[TMP41]])
; CHECK-NEXT:    [[TMP43:%.*]] = add i32 [[TMP42]], [[VEC_PHI]]
; CHECK-NEXT:    [[TMP44:%.*]] = select <4 x i1> [[TMP1]], <4 x i32> [[TMP40]], <4 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP45:%.*]] = call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> [[TMP44]])
; CHECK-NEXT:    [[TMP46]] = add i32 [[TMP45]], [[TMP43]]
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 4
; CHECK-NEXT:    [[TMP47:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[TMP47]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP8:![0-9]+]]
; CHECK:       middle.block:
; CHECK-NEXT:    [[CMP_N:%.*]] = icmp eq i64 [[N_VEC]], [[N]]
; CHECK-NEXT:    br i1 [[CMP_N]], label [[FOR_END:%.*]], label [[SCALAR_PH]]
; CHECK:       scalar.ph:
; CHECK-NEXT:    [[BC_RESUME_VAL:%.*]] = phi i64 [ [[N_VEC]], [[MIDDLE_BLOCK]] ], [ 0, [[ENTRY:%.*]] ]
; CHECK-NEXT:    [[BC_MERGE_RDX:%.*]] = phi i32 [ [[TMP46]], [[MIDDLE_BLOCK]] ], [ 5, [[ENTRY]] ]
; CHECK-NEXT:    br label [[FOR_BODY:%.*]]
; CHECK:       for.body:
; CHECK-NEXT:    [[IV:%.*]] = phi i64 [ [[IV_NEXT:%.*]], [[FOR_INC:%.*]] ], [ [[BC_RESUME_VAL]], [[SCALAR_PH]] ]
; CHECK-NEXT:    [[RDX:%.*]] = phi i32 [ [[RES:%.*]], [[FOR_INC]] ], [ [[BC_MERGE_RDX]], [[SCALAR_PH]] ]
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds i32, ptr [[COND]], i64 [[IV]]
; CHECK-NEXT:    [[TMP48:%.*]] = load i32, ptr [[ARRAYIDX]], align 4
; CHECK-NEXT:    [[TOBOOL_NOT:%.*]] = icmp eq i32 [[TMP48]], 0
; CHECK-NEXT:    br i1 [[TOBOOL_NOT]], label [[FOR_INC]], label [[IF_THEN:%.*]]
; CHECK:       if.then:
; CHECK-NEXT:    [[ARRAYIDX1:%.*]] = getelementptr inbounds i32, ptr [[A]], i64 [[IV]]
; CHECK-NEXT:    [[TMP49:%.*]] = load i32, ptr [[ARRAYIDX1]], align 4
; CHECK-NEXT:    [[ADD:%.*]] = add nsw i32 [[TMP49]], [[RDX]]
; CHECK-NEXT:    [[ARRAYIDX2:%.*]] = getelementptr inbounds i32, ptr [[B]], i64 [[IV]]
; CHECK-NEXT:    [[TMP50:%.*]] = load i32, ptr [[ARRAYIDX2]], align 4
; CHECK-NEXT:    [[ADD3:%.*]] = add nsw i32 [[ADD]], [[TMP50]]
; CHECK-NEXT:    br label [[FOR_INC]]
; CHECK:       for.inc:
; CHECK-NEXT:    [[RES]] = phi i32 [ [[ADD3]], [[IF_THEN]] ], [ [[RDX]], [[FOR_BODY]] ]
; CHECK-NEXT:    [[IV_NEXT]] = add nuw nsw i64 [[IV]], 1
; CHECK-NEXT:    [[EXITCOND_NOT:%.*]] = icmp eq i64 [[IV_NEXT]], [[N]]
; CHECK-NEXT:    br i1 [[EXITCOND_NOT]], label [[FOR_END]], label [[FOR_BODY]], !llvm.loop [[LOOP9:![0-9]+]]
; CHECK:       for.end:
; CHECK-NEXT:    [[RES_LCSSA:%.*]] = phi i32 [ [[RES]], [[FOR_INC]] ], [ [[TMP46]], [[MIDDLE_BLOCK]] ]
; CHECK-NEXT:    ret i32 [[RES_LCSSA]]
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.inc ], [ 0, %entry ]
  %rdx = phi i32 [ %res, %for.inc ], [ 5, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %cond, i64 %iv
  %0 = load i32, ptr %arrayidx
  %tobool.not = icmp eq i32 %0, 0
  br i1 %tobool.not, label %for.inc, label %if.then

if.then:
  %arrayidx1 = getelementptr inbounds i32, ptr %a, i64 %iv
  %1 = load i32, ptr %arrayidx1
  %add = add nsw i32 %1, %rdx
  %arrayidx2 = getelementptr inbounds i32, ptr %b, i64 %iv
  %2 = load i32, ptr %arrayidx2
  %add3 = add nsw i32 %add, %2
  br label %for.inc

for.inc:
  %res = phi i32 [ %add3, %if.then ], [ %rdx, %for.body ]
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:
  ret i32 %res
}

;
; Negative Tests
;

;
; Reduction not performed in loop as the phi has more than two incoming values
;
define i64 @nested_cond_and(ptr noalias nocapture readonly %a, ptr noalias nocapture readonly %b, ptr noalias nocapture readonly %cond, i64 %N){
; CHECK-LABEL: @nested_cond_and(
; CHECK:       vector.body:
; CHECK-NOT:     @llvm.vector.reduce.and
; CHECK:       middle.block:
; CHECK:         @llvm.vector.reduce.and
; CHECK:       scalar.ph
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %rdx = phi i64 [ 5, %entry ], [ %res, %for.inc ]
  %arrayidx = getelementptr inbounds i64, ptr %cond, i64 %iv
  %0 = load i64, ptr %arrayidx
  %tobool = icmp eq i64 %0, 0
  br i1 %tobool, label %if.then, label %for.inc

if.then:
  %arrayidx2 = getelementptr inbounds i64, ptr %a, i64 %iv
  %1 = load i64, ptr %arrayidx2
  %and1 = and i64 %rdx, %1
  %tobool2 = icmp eq i64 %1, 3
  br i1 %tobool2, label %if.then.2, label %for.inc

if.then.2:
  %arrayidx3 = getelementptr inbounds i64, ptr %b, i64 %iv
  %2 = load i64, ptr %arrayidx3
  %and2 = and i64 %rdx, %2
  br label %for.inc

for.inc:
  %res = phi i64 [ %and2, %if.then.2 ], [ %and1, %if.then ], [ %rdx, %for.body ]
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:
  ret i64 %res
}

; Chain of conditional & unconditional reductions. We currently only support conditional reductions
; if they are the last in the chain, i.e. the loop exit instruction is a Phi node. Therefore we reject
; the Phi (%rdx1) as it has more than one use.
;
define i32 @cond-uncond(ptr noalias %src1, ptr noalias %src2, ptr noalias %cond, i64 noundef %n) #0 {
; CHECK-LABEL: @cond-uncond(
; CHECK:       pred.load.continue6:
; CHECK-NOT:     @llvm.vector.reduce.add
; CHECK:       middle.block:
; CHECK:         @llvm.vector.reduce.add
; CHECK:       scalar.ph
entry:
  br label %for.body

for.body:
  %rdx1 = phi i32 [ %add2, %if.end ], [ 0, %entry ]
  %iv = phi i64 [ %iv.next, %if.end ], [ 0, %entry]
  %arrayidx = getelementptr inbounds i32, ptr %cond, i64 %iv
  %0 = load i32, ptr %arrayidx
  %tobool.not = icmp eq i32 %0, 0
  br i1 %tobool.not, label %if.end, label %if.then

if.then:
  %arrayidx1 = getelementptr inbounds i32, ptr %src2, i64 %iv
  %1 = load i32, ptr %arrayidx1
  %add = add nsw i32 %1, %rdx1
  br label %if.end

if.end:
  %res = phi i32 [ %add, %if.then ], [ %rdx1, %for.body ]
  %arrayidx2 = getelementptr inbounds i32, ptr %src1, i64 %iv
  %2 = load i32, ptr %arrayidx2
  %add2 = add nsw i32 %2, %res
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:
  ret i32 %add2
}

;
; Chain of two conditional reductions. We do not vectorise this with in-loop reductions as neither
; of the incoming values of the LoopExitInstruction (%res) is the reduction Phi (%rdx1).
;
define float @cond_cond(ptr noalias %src1, ptr noalias %src2, ptr noalias %cond, i64 %n) #0 {
; CHECK-LABEL: @cond_cond(
; CHECK:       pred.load.continue14:
; CHECK-NOT:     @llvm.vector.reduce.fadd
; CHECK:       middle.block:
; CHECK:         @llvm.vector.reduce.fadd
; CHECK:       scalar.ph
entry:
  br label %for.body

for.body:
  %rdx1 = phi float [ %res, %for.inc ], [ 2.000000e+00, %entry ]
  %iv = phi i64 [ %iv.next, %for.inc ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, ptr %cond, i64 %iv
  %0 = load float, ptr %arrayidx
  %cmp1 = fcmp fast oeq float %0, 3.000000e+00
  br i1 %cmp1, label %if.then, label %if.end

if.then:
  %arrayidx2 = getelementptr inbounds float, ptr %src1, i64 %iv
  %1 = load float, ptr %arrayidx2
  %add = fadd fast float %1, %rdx1
  br label %if.end

if.end:
  %rdx2 = phi float [ %add, %if.then ], [ %rdx1, %for.body ]
  %cmp5 = fcmp fast oeq float %0, 7.000000e+00
  br i1 %cmp5, label %if.then6, label %for.inc

if.then6:
  %arrayidx7 = getelementptr inbounds float, ptr %src2, i64 %iv
  %2 = load float, ptr %arrayidx7
  %add2 = fadd fast float %2, %rdx2
  br label %for.inc

for.inc:
  %res = phi float [ %add2, %if.then6 ], [ %rdx2, %if.end ]
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:
  ret float %res
}

;
; Chain of an unconditional & a conditional reduction. We do not vectorise this in-loop as neither of the
; incoming values of the LoopExitInstruction (%res) is the reduction Phi (%rdx).
;
define i32 @uncond_cond(ptr noalias %src1, ptr noalias %src2, ptr noalias %cond, i64 %N) #0 {
; CHECK-LABEL: @uncond_cond(
; CHECK:       pred.load.continue7:
; CHECK-NOT:     @llvm.vector.reduce.add
; CHECK:       middle.block:
; CHECK:         @llvm.vector.reduce.add
; CHECK:       scalar.ph
entry:
  br label %for.body

for.body:
  %rdx = phi i32 [ %res, %for.inc ], [ 0, %entry ]
  %iv = phi i64 [ %iv.next, %for.inc ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %src1, i64 %iv
  %0 = load i32, ptr %arrayidx
  %add1 = add nsw i32 %0, %rdx
  %arrayidx1 = getelementptr inbounds i32, ptr %cond, i64 %iv
  %1 = load i32, ptr %arrayidx1
  %tobool.not = icmp eq i32 %1, 0
  br i1 %tobool.not, label %for.inc, label %if.then

if.then:
  %arrayidx2 = getelementptr inbounds i32, ptr %src2, i64 %iv
  %2 = load i32, ptr %arrayidx2
  %add2 = add nsw i32 %2, %add1
  br label %for.inc

for.inc:
  %res = phi i32 [ %add2, %if.then ], [ %add1, %for.body ]
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:
  ret i32 %res
}

;
; Chain of multiple unconditional & conditional reductions. Does not vectorise in-loop as when we look back
; through the chain and check the number of uses of %add1, we find more than the expected one use.
;
define i32 @uncond_cond_uncond(ptr noalias %src1, ptr noalias %src2, ptr noalias %cond, i64 noundef %N) {
; CHECK-LABEL: @uncond_cond_uncond(
; CHECK:       pred.load.continue7:
; CHECK-NOT:     @llvm.vector.reduce.add
; CHECK:       middle.block:
; CHECK:         @llvm.vector.reduce.add
; CHECK:       scalar.ph
entry:
  br label %for.body

for.body:
  %rdx = phi i32 [ %add3, %if.end ], [ 0, %entry ]
  %iv = phi i64 [ %iv.next, %if.end ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %src1, i64 %iv
  %0 = load i32, ptr %arrayidx
  %add1 = add nsw i32 %0, %rdx
  %arrayidx1 = getelementptr inbounds i32, ptr %cond, i64 %iv
  %1 = load i32, ptr %arrayidx1
  %tobool.not = icmp eq i32 %1, 0
  br i1 %tobool.not, label %if.end, label %if.then

if.then:
  %arrayidx2 = getelementptr inbounds i32, ptr %src2, i64 %iv
  %2 = load i32, ptr %arrayidx2
  %add2 = add nsw i32 %2, %add1
  br label %if.end

if.end:
  %res = phi i32 [ %add2, %if.then ], [ %add1, %for.body ]
  %add3 = add nsw i32 %res, %0
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:
  ret i32 %add3
}
