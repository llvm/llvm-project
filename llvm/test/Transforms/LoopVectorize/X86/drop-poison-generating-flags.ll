; RUN: opt %s -passes=loop-vectorize -force-vector-width=4 -force-vector-interleave=1 -S | FileCheck %s

; Make sure that integer poison-generating flags (i.e., nuw/nsw, exact and inbounds)
; are dropped from instructions in blocks that need predication and are linearized
; and masked after vectorization. We only drop flags from scalar instructions that
; contribute to the address computation of a masked vector load/store. After
; linearizing the control flow and removing their guarding condition, these
; instructions could generate a poison value which would be used as base address of
; the masked vector load/store (see PR52111). For gather/scatter cases,
; posiong-generating flags can be preserved since poison addresses in the vector GEP
; reaching the gather/scatter instruction will be masked-out by the gather/scatter
; instruction itself and won't be used.
; We need AVX512 target features for the loop to be vectorized with masks instead of
; predicates.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; Drop poison-generating flags from 'sub' and 'getelementptr' feeding a masked load.
; Test for PR52111.
define void @drop_scalar_nuw_nsw(ptr noalias nocapture readonly %input,
                                 ptr %output) local_unnamed_addr #0 {
; CHECK-LABEL: @drop_scalar_nuw_nsw(
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, {{.*}} ]
; CHECK-NEXT:    [[VEC_IND:%.*]] = phi <4 x i64> [ <i64 0, i64 1, i64 2, i64 3>, {{.*}} ]
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[INDEX]], 0
; CHECK:         [[TMP4:%.*]] = icmp eq <4 x i64> [[VEC_IND]], zeroinitializer
; CHECK-NEXT:    [[TMP5:%.*]] = sub i64 [[TMP0]], 1
; CHECK-NEXT:    [[TMP6:%.*]] = getelementptr float, ptr [[INPUT:%.*]], i64 [[TMP5]]
; CHECK-NEXT:    [[TMP7:%.*]] = xor <4 x i1> [[TMP4]], <i1 true, i1 true, i1 true, i1 true>
; CHECK-NEXT:    [[TMP8:%.*]] = getelementptr float, ptr [[TMP6]], i32 0
; CHECK-NEXT:    [[WIDE_MASKED_LOAD:%.*]] = call <4 x float> @llvm.masked.load.v4f32.p0(ptr [[TMP8]], i32 4, <4 x i1> [[TMP7]], <4 x float> poison), !invariant.load !0
entry:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.inc, %if.end ]
  %i23 = icmp eq i64 %iv, 0
  br i1 %i23, label %if.end, label %if.then

if.then:
  %i27 = sub nuw nsw i64 %iv, 1
  %i29 = getelementptr inbounds float, ptr %input, i64 %i27
  %i30 = load float, ptr %i29, align 4, !invariant.load !0
  br label %if.end

if.end:
  %i34 = phi float [ 0.000000e+00, %loop.header ], [ %i30, %if.then ]
  %i35 = getelementptr inbounds float, ptr %output, i64 %iv
  store float %i34, ptr %i35, align 4
  %iv.inc = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.inc, 4
  br i1 %exitcond, label %loop.exit, label %loop.header

loop.exit:
  ret void
}

; Drop poison-generating flags from 'sub' and 'getelementptr' feeding a masked load.
; In this case, 'sub' and 'getelementptr' are not guarded by the predicate.
define void @drop_nonpred_scalar_nuw_nsw(ptr noalias nocapture readonly %input,
                                         ptr %output) local_unnamed_addr #0 {
; CHECK-LABEL: @drop_nonpred_scalar_nuw_nsw(
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, {{.*}} ]
; CHECK-NEXT:    [[VEC_IND:%.*]] = phi <4 x i64> [ <i64 0, i64 1, i64 2, i64 3>, {{.*}} ]
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[INDEX]], 0
; CHECK:         [[TMP5:%.*]] = sub i64 [[TMP0]], 1
; CHECK-NEXT:    [[TMP6:%.*]] = getelementptr float, ptr [[INPUT:%.*]], i64 [[TMP5]]
; CHECK-NEXT:    [[TMP4:%.*]] = icmp eq <4 x i64> [[VEC_IND]], zeroinitializer
; CHECK-NEXT:    [[TMP7:%.*]] = xor <4 x i1> [[TMP4]], <i1 true, i1 true, i1 true, i1 true>
; CHECK-NEXT:    [[TMP8:%.*]] = getelementptr float, ptr [[TMP6]], i32 0
; CHECK-NEXT:    [[WIDE_MASKED_LOAD:%.*]] = call <4 x float> @llvm.masked.load.v4f32.p0(ptr [[TMP8]], i32 4, <4 x i1> [[TMP7]], <4 x float> poison), !invariant.load !0
entry:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.inc, %if.end ]
  %i27 = sub i64 %iv, 1
  %i29 = getelementptr float, ptr %input, i64 %i27
  %i23 = icmp eq i64 %iv, 0
  br i1 %i23, label %if.end, label %if.then

if.then:
  %i30 = load float, ptr %i29, align 4, !invariant.load !0
  br label %if.end

if.end:
  %i34 = phi float [ 0.000000e+00, %loop.header ], [ %i30, %if.then ]
  %i35 = getelementptr inbounds float, ptr %output, i64 %iv
  store float %i34, ptr %i35, align 4
  %iv.inc = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.inc, 4
  br i1 %exitcond, label %loop.exit, label %loop.header

loop.exit:
  ret void
}

; Preserve poison-generating flags from vector 'sub', 'mul' and 'getelementptr' feeding a masked gather.
define void @preserve_vector_nuw_nsw(ptr noalias nocapture readonly %input,
                                     ptr %output) local_unnamed_addr #0 {
; CHECK-LABEL: @preserve_vector_nuw_nsw(
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, {{.*}} ]
; CHECK-NEXT:    [[VEC_IND:%.*]] = phi <4 x i64> [ <i64 0, i64 1, i64 2, i64 3>, {{.*}} ]
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[INDEX]], 0
; CHECK:         [[TMP4:%.*]] = icmp eq <4 x i64> [[VEC_IND]], zeroinitializer
; CHECK-NEXT:    [[TMP5:%.*]] = sub nuw nsw <4 x i64> [[VEC_IND]], <i64 1, i64 1, i64 1, i64 1>
; CHECK-NEXT:    [[TMP6:%.*]] = mul nuw nsw <4 x i64> [[TMP5]], <i64 2, i64 2, i64 2, i64 2>
; CHECK-NEXT:    [[TMP7:%.*]] = getelementptr inbounds float, ptr [[INPUT:%.*]], <4 x i64> [[TMP6]]
; CHECK-NEXT:    [[TMP8:%.*]] = xor <4 x i1> [[TMP4]], <i1 true, i1 true, i1 true, i1 true>
; CHECK-NEXT:    [[WIDE_MASKED_GATHER:%.*]] = call <4 x float> @llvm.masked.gather.v4f32.v4p0(<4 x ptr> [[TMP7]], i32 4, <4 x i1> [[TMP8]], <4 x float> poison), !invariant.load !0
entry:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.inc, %if.end ]
  %i23 = icmp eq i64 %iv, 0
  br i1 %i23, label %if.end, label %if.then

if.then:
  %i27 = sub nuw nsw i64 %iv, 1
  %i28 = mul nuw nsw i64 %i27, 2
  %i29 = getelementptr inbounds float, ptr %input, i64 %i28
  %i30 = load float, ptr %i29, align 4, !invariant.load !0
  br label %if.end

if.end:
  %i34 = phi float [ 0.000000e+00, %loop.header ], [ %i30, %if.then ]
  %i35 = getelementptr inbounds float, ptr %output, i64 %iv
  store float %i34, ptr %i35, align 4
  %iv.inc = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.inc, 4
  br i1 %exitcond, label %loop.exit, label %loop.header

loop.exit:
  ret void
}

; Drop poison-generating flags from vector 'sub' and 'gep' feeding a masked load.
define void @drop_vector_nuw_nsw(ptr noalias nocapture readonly %input,
                                 ptr %output, ptr noalias %ptrs) local_unnamed_addr #0 {
; CHECK-LABEL: @drop_vector_nuw_nsw(
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, {{.*}} ]
; CHECK-NEXT:    [[VEC_IND:%.*]] = phi <4 x i64> [ <i64 0, i64 1, i64 2, i64 3>, {{.*}} ]
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[INDEX]], 0
; CHECK:         [[TMP4:%.*]] = icmp eq <4 x i64> [[VEC_IND]], zeroinitializer
; CHECK-NEXT:    [[TMP5:%.*]] = getelementptr inbounds ptr, ptr [[PTRS:%.*]], i64 [[TMP0]]
; CHECK-NEXT:    [[TMP6:%.*]] = sub <4 x i64> [[VEC_IND]], <i64 1, i64 1, i64 1, i64 1>
; CHECK-NEXT:    [[TMP7:%.*]] = getelementptr float, ptr [[INPUT:%.*]], <4 x i64> [[TMP6]]
; CHECK:         [[TMP10:%.*]] = xor <4 x i1> [[TMP4]], <i1 true, i1 true, i1 true, i1 true>
; CHECK-NEXT:    [[TMP11:%.*]] = extractelement <4 x ptr> [[TMP7]], i32 0
; CHECK-NEXT:    [[TMP12:%.*]] = getelementptr float, ptr [[TMP11]], i32 0
; CHECK-NEXT:    [[WIDE_MASKED_LOAD:%.*]] = call <4 x float> @llvm.masked.load.v4f32.p0(ptr [[TMP12]], i32 4, <4 x i1> [[TMP10]], <4 x float> poison), !invariant.load !0
entry:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.inc, %if.end ]
  %i23 = icmp eq i64 %iv, 0
  %gep = getelementptr inbounds ptr, ptr %ptrs, i64 %iv
  %i27 = sub nuw nsw i64 %iv, 1
  %i29 = getelementptr inbounds float, ptr %input, i64 %i27
  store ptr %i29, ptr %gep
  br i1 %i23, label %if.end, label %if.then

if.then:
  %i30 = load float, ptr %i29, align 4, !invariant.load !0
  br label %if.end

if.end:
  %i34 = phi float [ 0.000000e+00, %loop.header ], [ %i30, %if.then ]
  %i35 = getelementptr inbounds float, ptr %output, i64 %iv
  store float %i34, ptr %i35, align 4
  %iv.inc = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.inc, 4
  br i1 %exitcond, label %loop.exit, label %loop.header

loop.exit:
  ret void
}

; Preserve poison-generating flags from 'sub', which is not contributing to any address computation
; of any masked load/store/gather/scatter.
define void @preserve_nuw_nsw_no_addr(ptr %output) local_unnamed_addr #0 {
; CHECK-LABEL: @preserve_nuw_nsw_no_addr(
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, {{.*}} ]
; CHECK-NEXT:    [[VEC_IND:%.*]] = phi <4 x i64> [ <i64 0, i64 1, i64 2, i64 3>, {{.*}} ]
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[INDEX]], 0
; CHECK:         [[TMP4:%.*]] = icmp eq <4 x i64> [[VEC_IND]], zeroinitializer
; CHECK-NEXT:    [[TMP5:%.*]] = sub nuw nsw <4 x i64> [[VEC_IND]], <i64 1, i64 1, i64 1, i64 1>
; CHECK-NEXT:    [[TMP6:%.*]] = xor <4 x i1> [[TMP4]], <i1 true, i1 true, i1 true, i1 true>
; CHECK-NEXT:    [[PREDPHI:%.*]] = select <4 x i1> [[TMP6]], <4 x i64> [[TMP5]], <4 x i64> zeroinitializer
; CHECK-NEXT:    [[TMP7:%.*]] = getelementptr inbounds i64, ptr [[OUTPUT:%.*]], i64 [[TMP0]]
; CHECK-NEXT:    [[TMP8:%.*]] = getelementptr inbounds i64, ptr [[TMP7]], i32 0
; CHECK-NEXT:    store <4 x i64> [[PREDPHI]], ptr [[TMP8]], align 4
entry:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.inc, %if.end ]
  %i23 = icmp eq i64 %iv, 0
  br i1 %i23, label %if.end, label %if.then

if.then:
  %i27 = sub nuw nsw i64 %iv, 1
  br label %if.end

if.end:
  %i34 = phi i64 [ 0, %loop.header ], [ %i27, %if.then ]
  %i35 = getelementptr inbounds i64, ptr %output, i64 %iv
  store i64 %i34, ptr %i35, align 4
  %iv.inc = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.inc, 4
  br i1 %exitcond, label %loop.exit, label %loop.header

loop.exit:
  ret void
}

; Drop poison-generating flags from 'sdiv' and 'getelementptr' feeding a masked load.
define void @drop_scalar_exact(ptr noalias nocapture readonly %input,
                               ptr %output) local_unnamed_addr #0 {
; CHECK-LABEL: @drop_scalar_exact(
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, {{.*}} ]
; CHECK-NEXT:    [[VEC_IND:%.*]] = phi <4 x i64> [ <i64 0, i64 1, i64 2, i64 3>, {{.*}} ]
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[INDEX]], 0
; CHECK:         [[TMP4:%.*]] = icmp ne <4 x i64> [[VEC_IND]], zeroinitializer
; CHECK-NEXT:    [[TMP5:%.*]] = and <4 x i64> [[VEC_IND]], <i64 1, i64 1, i64 1, i64 1>
; CHECK-NEXT:    [[TMP6:%.*]] = icmp eq <4 x i64> [[TMP5]], zeroinitializer
; CHECK-NEXT:    [[TMP7:%.*]] = and <4 x i1> [[TMP4]], [[TMP6]]
; CHECK-NEXT:    [[TMP8:%.*]] = sdiv i64 [[TMP0]], 1
; CHECK-NEXT:    [[TMP9:%.*]] = getelementptr float, ptr [[INPUT:%.*]], i64 [[TMP8]]
; CHECK-NEXT:    [[TMP10:%.*]] = xor <4 x i1> [[TMP7]], <i1 true, i1 true, i1 true, i1 true>
; CHECK-NEXT:    [[TMP11:%.*]] = getelementptr float, ptr [[TMP9]], i32 0
; CHECK-NEXT:    [[WIDE_MASKED_LOAD:%.*]] = call <4 x float> @llvm.masked.load.v4f32.p0(ptr [[TMP11]], i32 4, <4 x i1> [[TMP10]], <4 x float> poison), !invariant.load !0
entry:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.inc, %if.end ]
  %i7 = icmp ne i64 %iv, 0
  %i8 = and i64 %iv, 1
  %i9 = icmp eq i64 %i8, 0
  %i10 = and i1 %i7, %i9
  br i1 %i10, label %if.end, label %if.then

if.then:
  %i26 = sdiv exact i64 %iv, 1
  %i29 = getelementptr inbounds float, ptr %input, i64 %i26
  %i30 = load float, ptr %i29, align 4, !invariant.load !0
  br label %if.end

if.end:
  %i34 = phi float [ 0.000000e+00, %loop.header ], [ %i30, %if.then ]
  %i35 = getelementptr inbounds float, ptr %output, i64 %iv
  store float %i34, ptr %i35, align 4
  %iv.inc = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.inc, 4
  br i1 %exitcond, label %loop.exit, label %loop.header

loop.exit:
  ret void
}

define void @drop_zext_nneg(ptr noalias %p, ptr noalias %p1) #0 {
; CHECK-LABEL: define void @drop_zext_nneg(
; CHECK-SAME: ptr noalias [[P:%.*]], ptr noalias [[P1:%.*]]) #[[ATTR0:[0-9]+]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 true, label [[SCALAR_PH:%.*]], label [[VECTOR_SCEVCHECK:%.*]]
; CHECK:       vector.scevcheck:
; CHECK-NEXT:    br i1 true, label [[SCALAR_PH]], label [[VECTOR_PH:%.*]]
; CHECK:       vector.ph:
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_IND:%.*]] = phi <4 x i32> [ <i32 0, i32 1, i32 2, i32 3>, [[VECTOR_PH]] ], [ [[VEC_IND_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP0:%.*]] = icmp eq <4 x i32> [[VEC_IND]], zeroinitializer
; CHECK-NEXT:    [[TMP1:%.*]] = zext <4 x i32> [[VEC_IND]] to <4 x i64>
; CHECK-NEXT:    [[TMP2:%.*]] = extractelement <4 x i64> [[TMP1]], i32 0
; CHECK-NEXT:    [[TMP3:%.*]] = getelementptr double, ptr [[P]], i64 [[TMP2]]
; CHECK-NEXT:    [[TMP4:%.*]] = getelementptr double, ptr [[TMP3]], i32 0
; CHECK-NEXT:    [[WIDE_MASKED_LOAD:%.*]] = call <4 x double> @llvm.masked.load.v4f64.p0(ptr [[TMP4]], i32 8, <4 x i1> [[TMP0]], <4 x double> poison)
; CHECK-NEXT:    [[TMP5:%.*]] = xor <4 x i1> [[TMP0]], <i1 true, i1 true, i1 true, i1 true>
; CHECK-NEXT:    [[PREDPHI:%.*]] = select <4 x i1> [[TMP5]], <4 x double> zeroinitializer, <4 x double> [[WIDE_MASKED_LOAD]]
; CHECK-NEXT:    [[TMP6:%.*]] = extractelement <4 x double> [[PREDPHI]], i32 3
; CHECK-NEXT:    store double [[TMP6]], ptr [[P1]], align 8
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 4
; CHECK-NEXT:    [[VEC_IND_NEXT]] = add <4 x i32> [[VEC_IND]], <i32 4, i32 4, i32 4, i32 4>
; CHECK-NEXT:    [[TMP7:%.*]] = icmp eq i64 [[INDEX_NEXT]], 0
; CHECK-NEXT:    br i1 [[TMP7]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP17:![0-9]+]]
; CHECK:       middle.block:
; CHECK-NEXT:    br i1 true, label [[EXIT:%.*]], label [[SCALAR_PH]]
; CHECK:       scalar.ph:
; CHECK-NEXT:    [[BC_RESUME_VAL:%.*]] = phi i64 [ 0, [[MIDDLE_BLOCK]] ], [ 0, [[ENTRY:%.*]] ], [ 0, [[VECTOR_SCEVCHECK]] ]
; CHECK-NEXT:    br label [[BODY:%.*]]
; CHECK:       body:
; CHECK-NEXT:    [[IV:%.*]] = phi i64 [ [[NEXT:%.*]], [[ELSE:%.*]] ], [ [[BC_RESUME_VAL]], [[SCALAR_PH]] ]
; CHECK-NEXT:    [[TMP8:%.*]] = trunc i64 [[IV]] to i32
; CHECK-NEXT:    [[C:%.*]] = icmp eq i32 [[TMP8]], 0
; CHECK-NEXT:    br i1 [[C]], label [[THEN:%.*]], label [[ELSE]]
; CHECK:       then:
; CHECK-NEXT:    [[ZEXT:%.*]] = zext nneg i32 [[TMP8]] to i64
; CHECK-NEXT:    [[IDX1:%.*]] = getelementptr double, ptr [[P]], i64 [[ZEXT]]
; CHECK-NEXT:    [[IDX2:%.*]] = getelementptr double, ptr [[P]], i64 [[ZEXT]]
; CHECK-NEXT:    [[TMP9:%.*]] = load double, ptr [[IDX2]], align 8
; CHECK-NEXT:    br label [[ELSE]]
; CHECK:       else:
; CHECK-NEXT:    [[PHI:%.*]] = phi double [ [[TMP9]], [[THEN]] ], [ 0.000000e+00, [[BODY]] ]
; CHECK-NEXT:    store double [[PHI]], ptr [[P1]], align 8
; CHECK-NEXT:    [[NEXT]] = add i64 [[IV]], 1
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i64 [[NEXT]], 0
; CHECK-NEXT:    br i1 [[CMP]], label [[EXIT]], label [[BODY]], !llvm.loop [[LOOP18:![0-9]+]]
; CHECK:       exit:
; CHECK-NEXT:    ret void
;
entry:
  br label %body

body:
  %iv = phi i64 [ %next, %else ], [ 0, %entry ]
  %0 = trunc i64 %iv to i32
  %c = icmp eq i32 %0, 0
  br i1 %c, label %then, label %else

then:
  %zext = zext nneg i32 %0 to i64
  %idx1 = getelementptr double, ptr %p, i64 %zext
  %idx2 = getelementptr double, ptr %p, i64 %zext
  %1 = load double, ptr %idx2, align 8
  br label %else

else:
  %phi = phi double [ %1, %then ], [ 0.000000e+00, %body ]
  store double %phi, ptr %p1, align 8
  %next = add i64 %iv, 1
  %cmp = icmp eq i64 %next, 0
  br i1 %cmp, label %exit, label %body

exit:
  ret void
}

; Preserve poison-generating flags from 'sdiv' and 'getelementptr' feeding a masked gather.
define void @preserve_vector_exact_no_addr(ptr noalias nocapture readonly %input,
                                           ptr %output) local_unnamed_addr #0 {
; CHECK-LABEL: @preserve_vector_exact_no_addr(
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, {{.*}} ]
; CHECK-NEXT:    [[VEC_IND:%.*]] = phi <4 x i64> [ <i64 0, i64 1, i64 2, i64 3>, {{.*}} ]
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[INDEX]], 0
; CHECK:         [[TMP4:%.*]] = icmp ne <4 x i64> [[VEC_IND]], zeroinitializer
; CHECK-NEXT:    [[TMP5:%.*]] = and <4 x i64> [[VEC_IND]], <i64 1, i64 1, i64 1, i64 1>
; CHECK-NEXT:    [[TMP6:%.*]] = icmp eq <4 x i64> [[TMP5]], zeroinitializer
; CHECK-NEXT:    [[TMP7:%.*]] = and <4 x i1> [[TMP4]], [[TMP6]]
; CHECK-NEXT:    [[TMP8:%.*]] = sdiv exact <4 x i64> [[VEC_IND]], <i64 2, i64 2, i64 2, i64 2>
; CHECK-NEXT:    [[TMP9:%.*]] = getelementptr inbounds float, ptr [[INPUT:%.*]], <4 x i64> [[TMP8]]
; CHECK-NEXT:    [[TMP10:%.*]] = xor <4 x i1> [[TMP7]], <i1 true, i1 true, i1 true, i1 true>
; CHECK-NEXT:    [[WIDE_MASKED_GATHER:%.*]] = call <4 x float> @llvm.masked.gather.v4f32.v4p0(<4 x ptr> [[TMP9]], i32 4, <4 x i1> [[TMP10]], <4 x float> poison), !invariant.load !0
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.inc, %if.end ]
  %i7 = icmp ne i64 %iv, 0
  %i8 = and i64 %iv, 1
  %i9 = icmp eq i64 %i8, 0
  %i10 = and i1 %i7, %i9
  br i1 %i10, label %if.end, label %if.then

if.then:
  %i26 = sdiv exact i64 %iv, 2
  %i29 = getelementptr inbounds float, ptr %input, i64 %i26
  %i30 = load float, ptr %i29, align 4, !invariant.load !0
  br label %if.end

if.end:
  %i34 = phi float [ 0.000000e+00, %loop.header ], [ %i30, %if.then ]
  %i35 = getelementptr inbounds float, ptr %output, i64 %iv
  store float %i34, ptr %i35, align 4
  %iv.inc = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.inc, 4
  br i1 %exitcond, label %loop.exit, label %loop.header

loop.exit:
  ret void
}

; Preserve poison-generating flags from 'sdiv', which is not contributing to any address computation
; of any masked load/store/gather/scatter.
define void @preserve_exact_no_addr(ptr %output) local_unnamed_addr #0 {
; CHECK-LABEL: @preserve_exact_no_addr(
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, {{.*}} ]
; CHECK-NEXT:    [[VEC_IND:%.*]] = phi <4 x i64> [ <i64 0, i64 1, i64 2, i64 3>, {{.*}} ]
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[INDEX]], 0
; CHECK:         [[TMP4:%.*]] = icmp eq <4 x i64> [[VEC_IND]], zeroinitializer
; CHECK-NEXT:    [[TMP5:%.*]] = sdiv exact <4 x i64> [[VEC_IND]], <i64 2, i64 2, i64 2, i64 2>
; CHECK-NEXT:    [[TMP6:%.*]] = xor <4 x i1> [[TMP4]], <i1 true, i1 true, i1 true, i1 true>
; CHECK-NEXT:    [[PREDPHI:%.*]] = select <4 x i1> [[TMP6]], <4 x i64> [[TMP5]], <4 x i64> zeroinitializer
; CHECK-NEXT:    [[TMP7:%.*]] = getelementptr inbounds i64, ptr [[OUTPUT:%.*]], i64 [[TMP0]]
; CHECK-NEXT:    [[TMP8:%.*]] = getelementptr inbounds i64, ptr [[TMP7]], i32 0
; CHECK-NEXT:    store <4 x i64> [[PREDPHI]], ptr [[TMP8]], align 4
entry:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.inc, %if.end ]
  %i23 = icmp eq i64 %iv, 0
  br i1 %i23, label %if.end, label %if.then

if.then:
  %i27 = sdiv exact i64 %iv, 2
  br label %if.end

if.end:
  %i34 = phi i64 [ 0, %loop.header ], [ %i27, %if.then ]
  %i35 = getelementptr inbounds i64, ptr %output, i64 %iv
  store i64 %i34, ptr %i35, align 4
  %iv.inc = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.inc, 4
  br i1 %exitcond, label %loop.exit, label %loop.header

loop.exit:
  ret void
}

; Make sure we don't vectorize a loop with a phi feeding a poison value to
; a masked load/gather.
define void @dont_vectorize_poison_phi(ptr noalias nocapture readonly %input,
; CHECK-LABEL: @dont_vectorize_poison_phi(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[LOOP_HEADER:%.*]]
; CHECK:       loop.header:
; CHECK-NEXT:    [[POISON:%.*]] = phi i64 [ poison, [[ENTRY:%.*]] ], [ [[IV_INC:%.*]], [[IF_END:%.*]] ]
; CHECK-NEXT:    [[IV:%.*]] = phi i64 [ 0, [[ENTRY]] ], [ [[IV_INC]], [[IF_END]] ]
; CHECK-NEXT:    [[I23:%.*]] = icmp eq i64 [[IV]], 0
; CHECK-NEXT:    br i1 [[I23]], label [[IF_END]], label [[IF_THEN:%.*]]
; CHECK:       if.then:
; CHECK-NEXT:    [[I29:%.*]] = getelementptr inbounds float, ptr [[INPUT:%.*]], i64 [[POISON]]
; CHECK-NEXT:    [[I30:%.*]] = load float, ptr [[I29]], align 4, !invariant.load !0
; CHECK-NEXT:    br label [[IF_END]]
; CHECK:       if.end:
; CHECK-NEXT:    [[I34:%.*]] = phi float [ 0.000000e+00, [[LOOP_HEADER]] ], [ [[I30]], [[IF_THEN]] ]
; CHECK-NEXT:    [[I35:%.*]] = getelementptr inbounds float, ptr [[OUTPUT:%.*]], i64 [[IV]]
; CHECK-NEXT:    store float [[I34]], ptr [[I35]], align 4
; CHECK-NEXT:    [[IV_INC]] = add nuw nsw i64 [[IV]], 1
; CHECK-NEXT:    [[EXITCOND:%.*]] = icmp eq i64 [[IV_INC]], 4
; CHECK-NEXT:    br i1 [[EXITCOND]], label [[LOOP_EXIT:%.*]], label [[LOOP_HEADER]]
; CHECK:       loop.exit:
; CHECK-NEXT:    ret void
;
  ptr %output) local_unnamed_addr #0 {
entry:
  br label %loop.header

loop.header:
  %poison = phi i64 [ poison, %entry ], [ %iv.inc, %if.end ]
  %iv = phi i64 [ 0, %entry ], [ %iv.inc, %if.end ]
  %i23 = icmp eq i64 %iv, 0
  br i1 %i23, label %if.end, label %if.then

if.then:
  %i29 = getelementptr inbounds float, ptr %input, i64 %poison
  %i30 = load float, ptr %i29, align 4, !invariant.load !0
  br label %if.end

if.end:
  %i34 = phi float [ 0.000000e+00, %loop.header ], [ %i30, %if.then ]
  %i35 = getelementptr inbounds float, ptr %output, i64 %iv
  store float %i34, ptr %i35, align 4
  %iv.inc = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.inc, 4
  br i1 %exitcond, label %loop.exit, label %loop.header

loop.exit:
  ret void
}

@c = external global [5 x i8]

; Test case for https://github.com/llvm/llvm-project/issues/70590.
; Note that the then block has UB, but I could not find any other way to
; construct a suitable test case.
define void @pr70590_recipe_without_underlying_instr(i64 %n, ptr noalias %dst) {
; CHECK-LABEL: @pr70590_recipe_without_underlying_instr(
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH:%.+]] ], [ [[INDEX_NEXT:%.*]], [[PRED_SREM_CONTINUE6:%.*]] ]
; CHECK-NEXT:    [[VEC_IND:%.*]] = phi <4 x i64> [ <i64 0, i64 1, i64 2, i64 3>, [[VECTOR_PH]] ], [ [[VEC_IND_NEXT:%.*]], [[PRED_SREM_CONTINUE6]] ]
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP1:%.*]] = icmp eq <4 x i64> [[VEC_IND]],
; CHECK-NEXT:    [[TMP2:%.*]] = xor <4 x i1> [[TMP1]], <i1 true, i1 true, i1 true, i1 true>
; CHECK-NEXT:    [[TMP3:%.*]] = extractelement <4 x i1> [[TMP2]], i32 0
; CHECK-NEXT:    br i1 [[TMP3]], label [[PRED_SREM_IF:%.*]], label [[PRED_SREM_CONTINUE:%.*]]
; CHECK:       pred.srem.if:
; CHECK-NEXT:    [[TMP4:%.*]] = srem i64 3, 0
; CHECK-NEXT:    br label [[PRED_SREM_CONTINUE]]
; CHECK:       pred.srem.continue:
; CHECK-NEXT:    [[TMP5:%.*]] = phi i64 [ poison, %vector.body ], [ [[TMP4]], [[PRED_SREM_IF]] ]
; CHECK-NEXT:    [[TMP6:%.*]] = extractelement <4 x i1> [[TMP2]], i32 1
; CHECK-NEXT:    br i1 [[TMP6]], label [[PRED_SREM_IF1:%.*]], label [[PRED_SREM_CONTINUE2:%.*]]
; CHECK:       pred.srem.if1:
; CHECK-NEXT:    [[TMP7:%.*]] = srem i64 3, 0
; CHECK-NEXT:    br label [[PRED_SREM_CONTINUE2]]
; CHECK:       pred.srem.continue2:
; CHECK-NEXT:    [[TMP8:%.*]] = phi i64 [ poison, [[PRED_SREM_CONTINUE]] ], [ [[TMP7]], [[PRED_SREM_IF1]] ]
; CHECK-NEXT:    [[TMP9:%.*]] = extractelement <4 x i1> [[TMP2]], i32 2
; CHECK-NEXT:    br i1 [[TMP9]], label [[PRED_SREM_IF3:%.*]], label [[PRED_SREM_CONTINUE4:%.*]]
; CHECK:       pred.srem.if3:
; CHECK-NEXT:    [[TMP10:%.*]] = srem i64 3, 0
; CHECK-NEXT:    br label [[PRED_SREM_CONTINUE4]]
; CHECK:       pred.srem.continue4:
; CHECK-NEXT:    [[TMP11:%.*]] = phi i64 [ poison, [[PRED_SREM_CONTINUE2]] ], [ [[TMP10]], [[PRED_SREM_IF3]] ]
; CHECK-NEXT:    [[TMP12:%.*]] = extractelement <4 x i1> [[TMP2]], i32 3
; CHECK-NEXT:    br i1 [[TMP12]], label [[PRED_SREM_IF5:%.*]], label [[PRED_SREM_CONTINUE6]]
; CHECK:       pred.srem.if5:
; CHECK-NEXT:    [[TMP13:%.*]] = srem i64 3, 0
; CHECK-NEXT:    br label [[PRED_SREM_CONTINUE6]]
; CHECK:       pred.srem.continue6:
; CHECK-NEXT:    [[TMP14:%.*]] = phi i64 [ poison, [[PRED_SREM_CONTINUE4]] ], [ [[TMP13]], [[PRED_SREM_IF5]] ]
; CHECK-NEXT:    [[TMP15:%.*]] = add i64 [[TMP5]], -3
; CHECK-NEXT:    [[TMP16:%.*]] = add i64 [[TMP0]], [[TMP15]]
; CHECK-NEXT:    [[TMP17:%.*]] = getelementptr [5 x i8], ptr @c, i64 0, i64 [[TMP16]]
; CHECK-NEXT:    [[TMP18:%.*]] = getelementptr i8, ptr [[TMP17]], i32 0
; CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <4 x i8>, ptr [[TMP18]], align 1
; CHECK-NEXT:    [[PREDPHI:%.*]] = select <4 x i1> [[TMP2]], <4 x i8> [[WIDE_LOAD]], <4 x i8> zeroinitializer
; CHECK-NEXT:    [[TMP19:%.*]] = getelementptr i8, ptr %dst, i64 [[TMP0]]
; CHECK-NEXT:    [[TMP20:%.*]] = getelementptr i8, ptr [[TMP19]], i32 0
; CHECK-NEXT:    store <4 x i8> [[PREDPHI]], ptr [[TMP20]], align 4
; CHECK-NEXT:    [[VEC_IND_NEXT]] = add <4 x i64> [[VEC_IND]], <i64 4, i64 4, i64 4, i64 4>
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 4
; CHECK-NEXT:    br i1 true, label %middle.block, label %vector.body
; CHECK:       middle.block:

entry:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %inc, %loop.latch ]
  %cmp = icmp eq i64 %iv, %n
  br i1 %cmp, label %loop.latch, label %then

then:
  %rem = srem i64 3, 0
  %add3 = add i64 %rem, -3
  %add5 = add i64 %iv, %add3
  %gep = getelementptr [5 x i8], ptr @c, i64 0, i64 %add5
  %l = load i8, ptr %gep, align 1
  br label %loop.latch

loop.latch:
  %sr = phi i8 [ 0, %loop.header ], [ %l , %then ]
  %gep.dst = getelementptr i8, ptr %dst, i64 %iv
  store i8 %sr, ptr %gep.dst, align 4
  %inc = add i64 %iv, 1
  %exitcond.not = icmp eq i64 %inc, 4
  br i1 %exitcond.not, label %exit, label %loop.header

exit:
  ret void
}

attributes #0 = { noinline nounwind uwtable "target-features"="+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512vl" }

!0 = !{}
