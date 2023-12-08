; RUN: opt -passes=loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -S \
; RUN:   < %s | FileCheck %s
; RUN: opt -passes=loop-vectorize -force-vector-interleave=1 -force-vector-width=4 \
; RUN:   -scalable-vectorization=on -S < %s | FileCheck %s -check-prefix=SCALABLE

target triple = "riscv64"

define i32 @select_icmp(i32 %x, i32 %y, ptr nocapture readonly %c, i64 %n) #0 {
; CHECK-LABEL: @select_icmp
; CHECK:       vector.ph:
; CHECK-NEXT:    [[N_MOD_VF:%.*]] = urem i64 %n, 4
; CHECK-NEXT:    [[N_VEC:%.*]] = sub i64 %n, [[N_MOD_VF]]
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <4 x i32> poison, i32 [[X:%.*]], i64 0
; CHECK-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <4 x i32> [[BROADCAST_SPLATINSERT]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT1:%.*]] = insertelement <4 x i32> poison, i32 [[Y:%.*]], i64 0
; CHECK-NEXT:    [[BROADCAST_SPLAT2:%.*]] = shufflevector <4 x i32> [[BROADCAST_SPLATINSERT1]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH:%.*]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_PHI:%.*]] = phi <4 x i32> [ zeroinitializer, [[VECTOR_PH]] ], [ [[TMP5:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds i32, ptr [[C:%.*]], i64 [[TMP0]]
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr inbounds i32, ptr [[TMP1]], i32 0
; CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <4 x i32>, ptr [[TMP2]], align 4
; CHECK-NEXT:    [[TMP4:%.*]] = icmp slt <4 x i32> [[WIDE_LOAD]], [[BROADCAST_SPLAT]]
; CHECK-NEXT:    [[TMP5]] = select <4 x i1> [[TMP4]], <4 x i32> [[VEC_PHI]], <4 x i32> [[BROADCAST_SPLAT2]]
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 4
; CHECK-NEXT:    [[TMP6:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[TMP6]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP0:![0-9]+]]
; CHECK:       middle.block:
; CHECK-NEXT:    [[RDX_SELECT_CMP:%.*]] = icmp ne <4 x i32> [[TMP5]], zeroinitializer
; CHECK-NEXT:    [[TMP7:%.*]] = call i1 @llvm.vector.reduce.or.v4i1(<4 x i1> [[RDX_SELECT_CMP]])
; CHECK-NEXT:    [[RDX_SELECT:%.*]] = select i1 [[TMP7]], i32 [[Y]], i32 0
;
; SCALABLE-LABEL: @select_icmp
; SCALABLE:       vector.ph:
; SCALABLE-NEXT:    [[TMP2:%.*]] = call i64 @llvm.vscale.i64()
; SCALABLE-NEXT:    [[TMP3:%.*]] = mul i64 [[TMP2]], 4
; SCALABLE-NEXT:    [[N_MOD_VF:%.*]] = urem i64 %n, [[TMP3]]
; SCALABLE-NEXT:    [[N_VEC:%.*]] = sub i64 %n, [[N_MOD_VF]]
; SCALABLE-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <vscale x 4 x i32> poison, i32 [[X:%.*]], i64 0
; SCALABLE-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <vscale x 4 x i32> [[BROADCAST_SPLATINSERT]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; SCALABLE-NEXT:    [[BROADCAST_SPLATINSERT1:%.*]] = insertelement <vscale x 4 x i32> poison, i32 [[Y:%.*]], i64 0
; SCALABLE-NEXT:    [[BROADCAST_SPLAT2:%.*]] = shufflevector <vscale x 4 x i32> [[BROADCAST_SPLATINSERT1]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; SCALABLE-NEXT:    br label [[VECTOR_BODY:%.*]]
; SCALABLE:       vector.body:
; SCALABLE-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH:%.*]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; SCALABLE-NEXT:    [[VEC_PHI:%.*]] = phi <vscale x 4 x i32> [ zeroinitializer, [[VECTOR_PH]] ], [ [[TMP9:%.*]], [[VECTOR_BODY]] ]
; SCALABLE-NEXT:    [[TMP4:%.*]] = add i64 [[INDEX]], 0
; SCALABLE-NEXT:    [[TMP5:%.*]] = getelementptr inbounds i32, ptr [[C:%.*]], i64 [[TMP4]]
; SCALABLE-NEXT:    [[TMP6:%.*]] = getelementptr inbounds i32, ptr [[TMP5]], i32 0
; SCALABLE-NEXT:    [[WIDE_LOAD:%.*]] = load <vscale x 4 x i32>, ptr [[TMP6]], align 4
; SCALABLE-NEXT:    [[TMP8:%.*]] = icmp slt <vscale x 4 x i32> [[WIDE_LOAD]], [[BROADCAST_SPLAT]]
; SCALABLE-NEXT:    [[TMP9]] = select <vscale x 4 x i1> [[TMP8]], <vscale x 4 x i32> [[VEC_PHI]], <vscale x 4 x i32> [[BROADCAST_SPLAT2]]
; SCALABLE-NEXT:    [[TMP10:%.*]] = call i64 @llvm.vscale.i64()
; SCALABLE-NEXT:    [[TMP11:%.*]] = mul i64 [[TMP10]], 4
; SCALABLE-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], [[TMP11]]
; SCALABLE-NEXT:    [[TMP12:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; SCALABLE-NEXT:    br i1 [[TMP12]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP0:![0-9]+]]
; SCALABLE:       middle.block:
; SCALABLE-NEXT:    [[RDX_SELECT_CMP:%.*]] = icmp ne <vscale x 4 x i32> [[TMP9]], zeroinitializer
; SCALABLE-NEXT:    [[TMP13:%.*]] = call i1 @llvm.vector.reduce.or.nxv4i1(<vscale x 4 x i1> [[RDX_SELECT_CMP]])
; SCALABLE-NEXT:    [[RDX_SELECT:%.*]] = select i1 [[TMP13]], i32 [[Y]], i32 0
;
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %a = phi i32 [ 0, %entry], [ %cond, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %c, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %cmp1 = icmp slt i32 %0, %x
  %cond = select i1 %cmp1, i32 %a, i32 %y
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:
  ret i32 %cond
}

define i32 @select_fcmp(float %x, i32 %y, ptr nocapture readonly %c, i64 %n) #0 {
; CHECK-LABEL: @select_fcmp
; CHECK:       vector.ph:
; CHECK-NEXT:    [[N_MOD_VF:%.*]] = urem i64 %n, 4
; CHECK-NEXT:    [[N_VEC:%.*]] = sub i64 %n, [[N_MOD_VF]]
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <4 x float> poison, float [[X:%.*]], i64 0
; CHECK-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <4 x float> [[BROADCAST_SPLATINSERT]], <4 x float> poison, <4 x i32> zeroinitializer
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT1:%.*]] = insertelement <4 x i32> poison, i32 [[Y:%.*]], i64 0
; CHECK-NEXT:    [[BROADCAST_SPLAT2:%.*]] = shufflevector <4 x i32> [[BROADCAST_SPLATINSERT1]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH:%.*]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_PHI:%.*]] = phi <4 x i32> [ zeroinitializer, [[VECTOR_PH]] ], [ [[TMP5:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds float, ptr [[C:%.*]], i64 [[TMP0]]
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr inbounds float, ptr [[TMP1]], i32 0
; CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <4 x float>, ptr [[TMP2]], align 4
; CHECK-NEXT:    [[TMP4:%.*]] = fcmp fast olt <4 x float> [[WIDE_LOAD]], [[BROADCAST_SPLAT]]
; CHECK-NEXT:    [[TMP5]] = select <4 x i1> [[TMP4]], <4 x i32> [[VEC_PHI]], <4 x i32> [[BROADCAST_SPLAT2]]
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 4
; CHECK-NEXT:    [[TMP6:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[TMP6]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP4:![0-9]+]]
; CHECK:       middle.block:
; CHECK-NEXT:    [[RDX_SELECT_CMP:%.*]] = icmp ne <4 x i32> [[TMP5]], zeroinitializer
; CHECK-NEXT:    [[TMP7:%.*]] = call i1 @llvm.vector.reduce.or.v4i1(<4 x i1> [[RDX_SELECT_CMP]])
; CHECK-NEXT:    [[RDX_SELECT:%.*]] = select i1 [[TMP7]], i32 [[Y]], i32 0
;
; SCALABLE-LABEL: @select_fcmp
; SCALABLE:       vector.ph:
; SCALABLE-NEXT:    [[TMP2:%.*]] = call i64 @llvm.vscale.i64()
; SCALABLE-NEXT:    [[TMP3:%.*]] = mul i64 [[TMP2]], 4
; SCALABLE-NEXT:    [[N_MOD_VF:%.*]] = urem i64 %n, [[TMP3]]
; SCALABLE-NEXT:    [[N_VEC:%.*]] = sub i64 %n, [[N_MOD_VF]]
; SCALABLE-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <vscale x 4 x float> poison, float [[X:%.*]], i64 0
; SCALABLE-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <vscale x 4 x float> [[BROADCAST_SPLATINSERT]], <vscale x 4 x float> poison, <vscale x 4 x i32> zeroinitializer
; SCALABLE-NEXT:    [[BROADCAST_SPLATINSERT1:%.*]] = insertelement <vscale x 4 x i32> poison, i32 [[Y:%.*]], i64 0
; SCALABLE-NEXT:    [[BROADCAST_SPLAT2:%.*]] = shufflevector <vscale x 4 x i32> [[BROADCAST_SPLATINSERT1]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; SCALABLE-NEXT:    br label [[VECTOR_BODY:%.*]]
; SCALABLE:       vector.body:
; SCALABLE-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH:%.*]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; SCALABLE-NEXT:    [[VEC_PHI:%.*]] = phi <vscale x 4 x i32> [ zeroinitializer, [[VECTOR_PH]] ], [ [[TMP9:%.*]], [[VECTOR_BODY]] ]
; SCALABLE-NEXT:    [[TMP4:%.*]] = add i64 [[INDEX]], 0
; SCALABLE-NEXT:    [[TMP5:%.*]] = getelementptr inbounds float, ptr [[C:%.*]], i64 [[TMP4]]
; SCALABLE-NEXT:    [[TMP6:%.*]] = getelementptr inbounds float, ptr [[TMP5]], i32 0
; SCALABLE-NEXT:    [[WIDE_LOAD:%.*]] = load <vscale x 4 x float>, ptr [[TMP6]], align 4
; SCALABLE-NEXT:    [[TMP8:%.*]] = fcmp fast olt <vscale x 4 x float> [[WIDE_LOAD]], [[BROADCAST_SPLAT]]
; SCALABLE-NEXT:    [[TMP9]] = select <vscale x 4 x i1> [[TMP8]], <vscale x 4 x i32> [[VEC_PHI]], <vscale x 4 x i32> [[BROADCAST_SPLAT2]]
; SCALABLE-NEXT:    [[TMP10:%.*]] = call i64 @llvm.vscale.i64()
; SCALABLE-NEXT:    [[TMP11:%.*]] = mul i64 [[TMP10]], 4
; SCALABLE-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], [[TMP11]]
; SCALABLE-NEXT:    [[TMP12:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; SCALABLE-NEXT:    br i1 [[TMP12]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP4:![0-9]+]]
; SCALABLE:       middle.block:
; SCALABLE-NEXT:    [[RDX_SELECT_CMP:%.*]] = icmp ne <vscale x 4 x i32> [[TMP9]], zeroinitializer
; SCALABLE-NEXT:    [[TMP13:%.*]] = call i1 @llvm.vector.reduce.or.nxv4i1(<vscale x 4 x i1> [[RDX_SELECT_CMP]])
; SCALABLE-NEXT:    [[RDX_SELECT:%.*]] = select i1 [[TMP13]], i32 [[Y]], i32 0
;
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %a = phi i32 [ 0, %entry], [ %cond, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %c, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %cmp1 = fcmp fast olt float %0, %x
  %cond = select i1 %cmp1, i32 %a, i32 %y
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:
  ret i32 %cond
}

define i32 @select_const_i32_from_icmp(ptr nocapture readonly %v, i64 %n) #0 {
; CHECK-LABEL: @select_const_i32_from_icmp
; CHECK:       vector.ph:
; CHECK-NEXT:    [[N_MOD_VF:%.*]] = urem i64 %n, 4
; CHECK-NEXT:    [[N_VEC:%.*]] = sub i64 %n, [[N_MOD_VF]]
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH:%.*]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_PHI:%.*]] = phi <4 x i32> [ <i32 3, i32 3, i32 3, i32 3>, [[VECTOR_PH]] ], [ [[TMP5:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds i32, ptr [[V:%.*]], i64 [[TMP0]]
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr inbounds i32, ptr [[TMP1]], i32 0
; CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <4 x i32>, ptr [[TMP2]], align 4
; CHECK-NEXT:    [[TMP4:%.*]] = icmp eq <4 x i32> [[WIDE_LOAD]], <i32 3, i32 3, i32 3, i32 3>
; CHECK-NEXT:    [[TMP5]] = select <4 x i1> [[TMP4]], <4 x i32> [[VEC_PHI]], <4 x i32> <i32 7, i32 7, i32 7, i32 7>
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 4
; CHECK-NEXT:    [[TMP6:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[TMP6]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP6:![0-9]+]]
; CHECK:       middle.block:
; CHECK-NEXT:    [[RDX_SELECT_CMP:%.*]] = icmp ne <4 x i32> [[TMP5]], <i32 3, i32 3, i32 3, i32 3>
; CHECK-NEXT:    [[TMP7:%.*]] = call i1 @llvm.vector.reduce.or.v4i1(<4 x i1> [[RDX_SELECT_CMP]])
; CHECK-NEXT:    [[RDX_SELECT:%.*]] = select i1 [[TMP7]], i32 7, i32 3
;
; SCALABLE-LABEL: @select_const_i32_from_icmp
; SCALABLE:       vector.ph:
; SCALABLE-NEXT:    [[TMP2:%.*]] = call i64 @llvm.vscale.i64()
; SCALABLE-NEXT:    [[TMP3:%.*]] = mul i64 [[TMP2]], 4
; SCALABLE-NEXT:    [[N_MOD_VF:%.*]] = urem i64 %n, [[TMP3]]
; SCALABLE-NEXT:    [[N_VEC:%.*]] = sub i64 %n, [[N_MOD_VF]]
; SCALABLE-NEXT:    br label [[VECTOR_BODY:%.*]]
; SCALABLE:       vector.body:
; SCALABLE-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH:%.*]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; SCALABLE-NEXT:    [[VEC_PHI:%.*]] = phi <vscale x 4 x i32> [ shufflevector (<vscale x 4 x i32> insertelement (<vscale x 4 x i32> poison, i32 3, i64 0), <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer), [[VECTOR_PH]] ], [ [[TMP9:%.*]], [[VECTOR_BODY]] ]
; SCALABLE-NEXT:    [[TMP4:%.*]] = add i64 [[INDEX]], 0
; SCALABLE-NEXT:    [[TMP5:%.*]] = getelementptr inbounds i32, ptr [[V:%.*]], i64 [[TMP4]]
; SCALABLE-NEXT:    [[TMP6:%.*]] = getelementptr inbounds i32, ptr [[TMP5]], i32 0
; SCALABLE-NEXT:    [[WIDE_LOAD:%.*]] = load <vscale x 4 x i32>, ptr [[TMP6]], align 4
; SCALABLE-NEXT:    [[TMP8:%.*]] = icmp eq <vscale x 4 x i32> [[WIDE_LOAD]], shufflevector (<vscale x 4 x i32> insertelement (<vscale x 4 x i32> poison, i32 3, i64 0), <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer)
; SCALABLE-NEXT:    [[TMP9]] = select <vscale x 4 x i1> [[TMP8]], <vscale x 4 x i32> [[VEC_PHI]], <vscale x 4 x i32> shufflevector (<vscale x 4 x i32> insertelement (<vscale x 4 x i32> poison, i32 7, i64 0), <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer)
; SCALABLE-NEXT:    [[TMP10:%.*]] = call i64 @llvm.vscale.i64()
; SCALABLE-NEXT:    [[TMP11:%.*]] = mul i64 [[TMP10]], 4
; SCALABLE-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], [[TMP11]]
; SCALABLE-NEXT:    [[TMP12:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; SCALABLE-NEXT:    br i1 [[TMP12]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP6:![0-9]+]]
; SCALABLE:       middle.block:
; SCALABLE-NEXT:    [[RDX_SELECT_CMP:%.*]] = icmp ne <vscale x 4 x i32> [[TMP9]], shufflevector (<vscale x 4 x i32> insertelement (<vscale x 4 x i32> poison, i32 3, i64 0), <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer)
; SCALABLE-NEXT:    [[TMP13:%.*]] = call i1 @llvm.vector.reduce.or.nxv4i1(<vscale x 4 x i1> [[RDX_SELECT_CMP]])
; SCALABLE-NEXT:    [[RDX_SELECT:%.*]] = select i1 [[TMP13]], i32 7, i32 3
;
entry:
  br label %for.body

for.body:                                      ; preds = %entry, %for.body
  %0 = phi i64 [ 0, %entry ], [ %6, %for.body ]
  %1 = phi i32 [ 3, %entry ], [ %5, %for.body ]
  %2 = getelementptr inbounds i32, ptr %v, i64 %0
  %3 = load i32, ptr %2, align 4
  %4 = icmp eq i32 %3, 3
  %5 = select i1 %4, i32 %1, i32 7
  %6 = add nuw nsw i64 %0, 1
  %7 = icmp eq i64 %6, %n
  br i1 %7, label %exit, label %for.body

exit:                                     ; preds = %for.body
  ret i32 %5
}

define i32 @select_i32_from_icmp(ptr nocapture readonly %v, i32 %a, i32 %b, i64 %n) #0 {
; CHECK-LABEL: @select_i32_from_icmp
; CHECK:       vector.ph:
; CHECK-NEXT:    [[N_MOD_VF:%.*]] = urem i64 %n, 4
; CHECK-NEXT:    [[N_VEC:%.*]] = sub i64 %n, [[N_MOD_VF]]
; CHECK-NEXT:    [[MINMAX_IDENT_SPLATINSERT:%.*]] = insertelement <4 x i32> poison, i32 [[A:%.*]], i64 0
; CHECK-NEXT:    [[MINMAX_IDENT_SPLAT:%.*]] = shufflevector <4 x i32> [[MINMAX_IDENT_SPLATINSERT]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <4 x i32> poison, i32 [[B:%.*]], i64 0
; CHECK-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <4 x i32> [[BROADCAST_SPLATINSERT]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH:%.*]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_PHI:%.*]] = phi <4 x i32> [ [[MINMAX_IDENT_SPLAT]], [[VECTOR_PH]] ], [ [[TMP5:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds i32, ptr [[V:%.*]], i64 [[TMP0]]
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr inbounds i32, ptr [[TMP1]], i32 0
; CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <4 x i32>, ptr [[TMP2]], align 4
; CHECK-NEXT:    [[TMP4:%.*]] = icmp eq <4 x i32> [[WIDE_LOAD]], <i32 3, i32 3, i32 3, i32 3>
; CHECK-NEXT:    [[TMP5]] = select <4 x i1> [[TMP4]], <4 x i32> [[VEC_PHI]], <4 x i32> [[BROADCAST_SPLAT]]
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 4
; CHECK-NEXT:    [[TMP6:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[TMP6]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP8:![0-9]+]]
; CHECK:       middle.block:
; CHECK-NEXT:    [[DOTSPLATINSERT:%.*]] = insertelement <4 x i32> poison, i32 [[A]], i64 0
; CHECK-NEXT:    [[DOTSPLAT:%.*]] = shufflevector <4 x i32> [[DOTSPLATINSERT]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK-NEXT:    [[RDX_SELECT_CMP:%.*]] = icmp ne <4 x i32> [[TMP5]], [[DOTSPLAT]]
; CHECK-NEXT:    [[TMP7:%.*]] = call i1 @llvm.vector.reduce.or.v4i1(<4 x i1> [[RDX_SELECT_CMP]])
; CHECK-NEXT:    [[RDX_SELECT:%.*]] = select i1 [[TMP7]], i32 [[B]], i32 [[A]]
;
; SCALABLE-LABEL: @select_i32_from_icmp
; SCALABLE:       vector.ph:
; SCALABLE-NEXT:    [[TMP2:%.*]] = call i64 @llvm.vscale.i64()
; SCALABLE-NEXT:    [[TMP3:%.*]] = mul i64 [[TMP2]], 4
; SCALABLE-NEXT:    [[N_MOD_VF:%.*]] = urem i64 %n, [[TMP3]]
; SCALABLE-NEXT:    [[N_VEC:%.*]] = sub i64 %n, [[N_MOD_VF]]
; SCALABLE-NEXT:    [[MINMAX_IDENT_SPLATINSERT:%.*]] = insertelement <vscale x 4 x i32> poison, i32 [[A:%.*]], i64 0
; SCALABLE-NEXT:    [[MINMAX_IDENT_SPLAT:%.*]] = shufflevector <vscale x 4 x i32> [[MINMAX_IDENT_SPLATINSERT]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; SCALABLE-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <vscale x 4 x i32> poison, i32 [[B:%.*]], i64 0
; SCALABLE-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <vscale x 4 x i32> [[BROADCAST_SPLATINSERT]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; SCALABLE-NEXT:    br label [[VECTOR_BODY:%.*]]
; SCALABLE:       vector.body:
; SCALABLE-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH:%.*]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; SCALABLE-NEXT:    [[VEC_PHI:%.*]] = phi <vscale x 4 x i32> [ [[MINMAX_IDENT_SPLAT]], [[VECTOR_PH]] ], [ [[TMP9:%.*]], [[VECTOR_BODY]] ]
; SCALABLE-NEXT:    [[TMP4:%.*]] = add i64 [[INDEX]], 0
; SCALABLE-NEXT:    [[TMP5:%.*]] = getelementptr inbounds i32, ptr [[V:%.*]], i64 [[TMP4]]
; SCALABLE-NEXT:    [[TMP6:%.*]] = getelementptr inbounds i32, ptr [[TMP5]], i32 0
; SCALABLE-NEXT:    [[WIDE_LOAD:%.*]] = load <vscale x 4 x i32>, ptr [[TMP6]], align 4
; SCALABLE-NEXT:    [[TMP8:%.*]] = icmp eq <vscale x 4 x i32> [[WIDE_LOAD]], shufflevector (<vscale x 4 x i32> insertelement (<vscale x 4 x i32> poison, i32 3, i64 0), <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer)
; SCALABLE-NEXT:    [[TMP9]] = select <vscale x 4 x i1> [[TMP8]], <vscale x 4 x i32> [[VEC_PHI]], <vscale x 4 x i32> [[BROADCAST_SPLAT]]
; SCALABLE-NEXT:    [[TMP10:%.*]] = call i64 @llvm.vscale.i64()
; SCALABLE-NEXT:    [[TMP11:%.*]] = mul i64 [[TMP10]], 4
; SCALABLE-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], [[TMP11]]
; SCALABLE-NEXT:    [[TMP12:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; SCALABLE-NEXT:    br i1 [[TMP12]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP8:![0-9]+]]
; SCALABLE:       middle.block:
; SCALABLE-NEXT:    [[DOTSPLATINSERT:%.*]] = insertelement <vscale x 4 x i32> poison, i32 [[A]], i64 0
; SCALABLE-NEXT:    [[DOTSPLAT:%.*]] = shufflevector <vscale x 4 x i32> [[DOTSPLATINSERT]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; SCALABLE-NEXT:    [[RDX_SELECT_CMP:%.*]] = icmp ne <vscale x 4 x i32> [[TMP9]], [[DOTSPLAT]]
; SCALABLE-NEXT:    [[TMP13:%.*]] = call i1 @llvm.vector.reduce.or.nxv4i1(<vscale x 4 x i1> [[RDX_SELECT_CMP]])
; SCALABLE-NEXT:    [[RDX_SELECT:%.*]] = select i1 [[TMP13]], i32 [[B]], i32 [[A]]
;
entry:
  br label %for.body

for.body:                                      ; preds = %entry, %for.body
  %0 = phi i64 [ 0, %entry ], [ %6, %for.body ]
  %1 = phi i32 [ %a, %entry ], [ %5, %for.body ]
  %2 = getelementptr inbounds i32, ptr %v, i64 %0
  %3 = load i32, ptr %2, align 4
  %4 = icmp eq i32 %3, 3
  %5 = select i1 %4, i32 %1, i32 %b
  %6 = add nuw nsw i64 %0, 1
  %7 = icmp eq i64 %6, %n
  br i1 %7, label %exit, label %for.body

exit:                                     ; preds = %for.body
  ret i32 %5
}

define i32 @select_const_i32_from_fcmp(ptr nocapture readonly %v, i64 %n) #0 {
; CHECK-LABEL: @select_const_i32_from_fcmp
; CHECK:       vector.ph:
; CHECK-NEXT:    [[N_MOD_VF:%.*]] = urem i64 %n, 4
; CHECK-NEXT:    [[N_VEC:%.*]] = sub i64 %n, [[N_MOD_VF]]
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH:%.*]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_PHI:%.*]] = phi <4 x i32> [ <i32 2, i32 2, i32 2, i32 2>, [[VECTOR_PH]] ], [ [[TMP5:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds float, ptr [[V:%.*]], i64 [[TMP0]]
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr inbounds float, ptr [[TMP1]], i32 0
; CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <4 x float>, ptr [[TMP2]], align 4
; CHECK-NEXT:    [[TMP4:%.*]] = fcmp fast ueq <4 x float> [[WIDE_LOAD]], <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
; CHECK-NEXT:    [[TMP5]] = select <4 x i1> [[TMP4]], <4 x i32> [[VEC_PHI]], <4 x i32> <i32 1, i32 1, i32 1, i32 1>
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 4
; CHECK-NEXT:    [[TMP6:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[TMP6]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP10:![0-9]+]]
; CHECK:       middle.block:
; CHECK-NEXT:    [[RDX_SELECT_CMP:%.*]] = icmp ne <4 x i32> [[TMP5]], <i32 2, i32 2, i32 2, i32 2>
; CHECK-NEXT:    [[TMP7:%.*]] = call i1 @llvm.vector.reduce.or.v4i1(<4 x i1> [[RDX_SELECT_CMP]])
; CHECK-NEXT:    [[RDX_SELECT:%.*]] = select i1 [[TMP7]], i32 1, i32 2
;
; SCALABLE-LABEL: @select_const_i32_from_fcmp
; SCALABLE:       vector.ph:
; SCALABLE-NEXT:    [[TMP2:%.*]] = call i64 @llvm.vscale.i64()
; SCALABLE-NEXT:    [[TMP3:%.*]] = mul i64 [[TMP2]], 4
; SCALABLE-NEXT:    [[N_MOD_VF:%.*]] = urem i64 %n, [[TMP3]]
; SCALABLE-NEXT:    [[N_VEC:%.*]] = sub i64 %n, [[N_MOD_VF]]
; SCALABLE-NEXT:    br label [[VECTOR_BODY:%.*]]
; SCALABLE:       vector.body:
; SCALABLE-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH:%.*]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; SCALABLE-NEXT:    [[VEC_PHI:%.*]] = phi <vscale x 4 x i32> [ shufflevector (<vscale x 4 x i32> insertelement (<vscale x 4 x i32> poison, i32 2, i64 0), <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer), [[VECTOR_PH]] ], [ [[TMP9:%.*]], [[VECTOR_BODY]] ]
; SCALABLE-NEXT:    [[TMP4:%.*]] = add i64 [[INDEX]], 0
; SCALABLE-NEXT:    [[TMP5:%.*]] = getelementptr inbounds float, ptr [[V:%.*]], i64 [[TMP4]]
; SCALABLE-NEXT:    [[TMP6:%.*]] = getelementptr inbounds float, ptr [[TMP5]], i32 0
; SCALABLE-NEXT:    [[WIDE_LOAD:%.*]] = load <vscale x 4 x float>, ptr [[TMP6]], align 4
; SCALABLE-NEXT:    [[TMP8:%.*]] = fcmp fast ueq <vscale x 4 x float> [[WIDE_LOAD]], shufflevector (<vscale x 4 x float> insertelement (<vscale x 4 x float> poison, float 3.000000e+00, i64 0), <vscale x 4 x float> poison, <vscale x 4 x i32> zeroinitializer)
; SCALABLE-NEXT:    [[TMP9]] = select <vscale x 4 x i1> [[TMP8]], <vscale x 4 x i32> [[VEC_PHI]], <vscale x 4 x i32> shufflevector (<vscale x 4 x i32> insertelement (<vscale x 4 x i32> poison, i32 1, i64 0), <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer)
; SCALABLE-NEXT:    [[TMP10:%.*]] = call i64 @llvm.vscale.i64()
; SCALABLE-NEXT:    [[TMP11:%.*]] = mul i64 [[TMP10]], 4
; SCALABLE-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], [[TMP11]]
; SCALABLE-NEXT:    [[TMP12:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; SCALABLE-NEXT:    br i1 [[TMP12]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP10:![0-9]+]]
; SCALABLE:       middle.block:
; SCALABLE-NEXT:    [[RDX_SELECT_CMP:%.*]] = icmp ne <vscale x 4 x i32> [[TMP9]], shufflevector (<vscale x 4 x i32> insertelement (<vscale x 4 x i32> poison, i32 2, i64 0), <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer)
; SCALABLE-NEXT:    [[TMP13:%.*]] = call i1 @llvm.vector.reduce.or.nxv4i1(<vscale x 4 x i1> [[RDX_SELECT_CMP]])
; SCALABLE-NEXT:    [[RDX_SELECT:%.*]] = select i1 [[TMP13]], i32 1, i32 2
;
entry:
  br label %for.body

for.body:                                      ; preds = %entry, %for.body
  %0 = phi i64 [ 0, %entry ], [ %6, %for.body ]
  %1 = phi i32 [ 2, %entry ], [ %5, %for.body ]
  %2 = getelementptr inbounds float, ptr %v, i64 %0
  %3 = load float, ptr %2, align 4
  %4 = fcmp fast ueq float %3, 3.0
  %5 = select i1 %4, i32 %1, i32 1
  %6 = add nuw nsw i64 %0, 1
  %7 = icmp eq i64 %6, %n
  br i1 %7, label %exit, label %for.body

exit:                                     ; preds = %for.body
  ret i32 %5
}

define float @select_const_f32_from_icmp(ptr nocapture readonly %v, i64 %n) #0 {
; CHECK-LABEL: @select_const_f32_from_icmp
; CHECK-NOT: vector.body
;
; SCALABLE-LABEL: @select_const_f32_from_icmp
; SCALABLE-NOT: vector.body
;
entry:
  br label %for.body

for.body:                                      ; preds = %entry, %for.body
  %0 = phi i64 [ 0, %entry ], [ %6, %for.body ]
  %1 = phi fast float [ 3.0, %entry ], [ %5, %for.body ]
  %2 = getelementptr inbounds i32, ptr %v, i64 %0
  %3 = load i32, ptr %2, align 4
  %4 = icmp eq i32 %3, 3
  %5 = select fast i1 %4, float %1, float 7.0
  %6 = add nuw nsw i64 %0, 1
  %7 = icmp eq i64 %6, %n
  br i1 %7, label %exit, label %for.body

exit:                                     ; preds = %for.body
  ret float %5
}

define i32 @pred_select_const_i32_from_icmp(ptr noalias nocapture readonly %src1, ptr noalias nocapture readonly %src2, i64 %n) #0 {
; CHECK-LABEL: @pred_select_const_i32_from_icmp
; CHECK:       vector.ph:
; CHECK-NEXT:    [[N_MOD_VF:%.*]] = urem i64 %n, 4
; CHECK-NEXT:    [[N_VEC:%.*]] = sub i64 %n, [[N_MOD_VF]]
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH:%.*]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_PHI:%.*]] = phi <4 x i32> [ zeroinitializer, [[VECTOR_PH]] ], [ [[PREDPHI:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds i32, ptr [[SRC1:%.*]], i64 [[TMP0]]
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr inbounds i32, ptr [[TMP1]], i32 0
; CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <4 x i32>, ptr [[TMP2]], align 4
; CHECK-NEXT:    [[TMP4:%.*]] = icmp sgt <4 x i32> [[WIDE_LOAD]], <i32 35, i32 35, i32 35, i32 35>
; CHECK-NEXT:    [[TMP5:%.*]] = getelementptr i32, ptr [[SRC2:%.*]], i64 [[TMP0]]
; CHECK-NEXT:    [[TMP6:%.*]] = getelementptr i32, ptr [[TMP5]], i32 0
; CHECK-NEXT:    [[WIDE_MASKED_LOAD:%.*]] = call <4 x i32> @llvm.masked.load.v4i32.p0(ptr [[TMP6]], i32 4, <4 x i1> [[TMP4]], <4 x i32> poison)
; CHECK-NEXT:    [[TMP8:%.*]] = icmp eq <4 x i32> [[WIDE_MASKED_LOAD]], <i32 2, i32 2, i32 2, i32 2>
; CHECK-NEXT:    [[TMP9:%.*]] = select <4 x i1> [[TMP8]], <4 x i32> <i32 1, i32 1, i32 1, i32 1>, <4 x i32> [[VEC_PHI]]
; CHECK-NEXT:    [[TMP10:%.*]] = xor <4 x i1> [[TMP4]], <i1 true, i1 true, i1 true, i1 true>
; CHECK-NEXT:    [[PREDPHI]] = select <4 x i1> [[TMP4]], <4 x i32> [[TMP9]], <4 x i32> [[VEC_PHI]]
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 4
; CHECK-NEXT:    [[TMP11:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[TMP11]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP12:![0-9]+]]
; CHECK:       middle.block:
; CHECK-NEXT:    [[RDX_SELECT_CMP:%.*]] = icmp ne <4 x i32> [[PREDPHI]], zeroinitializer
; CHECK-NEXT:    [[TMP12:%.*]] = call i1 @llvm.vector.reduce.or.v4i1(<4 x i1> [[RDX_SELECT_CMP]])
; CHECK-NEXT:    [[RDX_SELECT:%.*]] = select i1 [[TMP12]], i32 1, i32 0
;
; SCALABLE-LABEL: @pred_select_const_i32_from_icmp
; SCALABLE:       vector.ph:
; SCALABLE-NEXT:    [[TMP2:%.*]] = call i64 @llvm.vscale.i64()
; SCALABLE-NEXT:    [[TMP3:%.*]] = mul i64 [[TMP2]], 4
; SCALABLE-NEXT:    [[N_MOD_VF:%.*]] = urem i64 %n, [[TMP3]]
; SCALABLE-NEXT:    [[N_VEC:%.*]] = sub i64 %n, [[N_MOD_VF]]
; SCALABLE-NEXT:    br label [[VECTOR_BODY:%.*]]
; SCALABLE:       vector.body:
; SCALABLE-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH:%.*]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; SCALABLE-NEXT:    [[VEC_PHI:%.*]] = phi <vscale x 4 x i32> [ zeroinitializer, [[VECTOR_PH]] ], [ [[PREDPHI:%.*]], [[VECTOR_BODY]] ]
; SCALABLE-NEXT:    [[TMP4:%.*]] = add i64 [[INDEX]], 0
; SCALABLE-NEXT:    [[TMP5:%.*]] = getelementptr inbounds i32, ptr [[SRC1:%.*]], i64 [[TMP4]]
; SCALABLE-NEXT:    [[TMP6:%.*]] = getelementptr inbounds i32, ptr [[TMP5]], i32 0
; SCALABLE-NEXT:    [[WIDE_LOAD:%.*]] = load <vscale x 4 x i32>, ptr [[TMP6]], align 4
; SCALABLE-NEXT:    [[TMP8:%.*]] = icmp sgt <vscale x 4 x i32> [[WIDE_LOAD]], shufflevector (<vscale x 4 x i32> insertelement (<vscale x 4 x i32> poison, i32 35, i64 0), <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer)
; SCALABLE-NEXT:    [[TMP9:%.*]] = getelementptr i32, ptr [[SRC2:%.*]], i64 [[TMP4]]
; SCALABLE-NEXT:    [[TMP10:%.*]] = getelementptr i32, ptr [[TMP9]], i32 0
; SCALABLE-NEXT:    [[WIDE_MASKED_LOAD:%.*]] = call <vscale x 4 x i32> @llvm.masked.load.nxv4i32.p0(ptr [[TMP10]], i32 4, <vscale x 4 x i1> [[TMP8]], <vscale x 4 x i32> poison)
; SCALABLE-NEXT:    [[TMP12:%.*]] = icmp eq <vscale x 4 x i32> [[WIDE_MASKED_LOAD]], shufflevector (<vscale x 4 x i32> insertelement (<vscale x 4 x i32> poison, i32 2, i64 0), <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer)
; SCALABLE-NEXT:    [[TMP13:%.*]] = select <vscale x 4 x i1> [[TMP12]], <vscale x 4 x i32> shufflevector (<vscale x 4 x i32> insertelement (<vscale x 4 x i32> poison, i32 1, i64 0), <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x i32> [[VEC_PHI]]
; SCALABLE-NEXT:    [[TMP14:%.*]] = xor <vscale x 4 x i1> [[TMP8]], shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer)
; SCALABLE-NEXT:    [[PREDPHI]] = select <vscale x 4 x i1> [[TMP8]], <vscale x 4 x i32> [[TMP13]], <vscale x 4 x i32> [[VEC_PHI]]
; SCALABLE-NEXT:    [[TMP15:%.*]] = call i64 @llvm.vscale.i64()
; SCALABLE-NEXT:    [[TMP16:%.*]] = mul i64 [[TMP15]], 4
; SCALABLE-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], [[TMP16]]
; SCALABLE-NEXT:    [[TMP17:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; SCALABLE-NEXT:    br i1 [[TMP17]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP12:![0-9]+]]
; SCALABLE:       middle.block:
; SCALABLE-NEXT:    [[RDX_SELECT_CMP:%.*]] = icmp ne <vscale x 4 x i32> [[PREDPHI]], zeroinitializer
; SCALABLE-NEXT:    [[TMP18:%.*]] = call i1 @llvm.vector.reduce.or.nxv4i1(<vscale x 4 x i1> [[RDX_SELECT_CMP]])
; SCALABLE-NEXT:    [[RDX_SELECT:%.*]] = select i1 [[TMP18]], i32 1, i32 0
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.inc
  %i.013 = phi i64 [ %inc, %for.inc ], [ 0, %entry ]
  %r.012 = phi i32 [ %r.1, %for.inc ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %src1, i64 %i.013
  %0 = load i32, ptr %arrayidx, align 4
  %cmp1 = icmp sgt i32 %0, 35
  br i1 %cmp1, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %arrayidx2 = getelementptr inbounds i32, ptr %src2, i64 %i.013
  %1 = load i32, ptr %arrayidx2, align 4
  %cmp3 = icmp eq i32 %1, 2
  %spec.select = select i1 %cmp3, i32 1, i32 %r.012
  br label %for.inc

for.inc:                                          ; preds = %if.then, %for.body
  %r.1 = phi i32 [ %r.012, %for.body ], [ %spec.select, %if.then ]
  %inc = add nuw nsw i64 %i.013, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %for.end.loopexit, label %for.body

for.end.loopexit:                                 ; preds = %for.inc
  %r.1.lcssa = phi i32 [ %r.1, %for.inc ]
  ret i32 %r.1.lcssa
}

attributes #0 = { "target-features"="+f,+v" }
