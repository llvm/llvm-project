; RUN: opt -mcpu=skx -S -loop-vectorize -instcombine -simplifycfg -force-vector-width=8 -force-vector-interleave=1 -enable-interleaved-mem-accesses < %s | FileCheck %s -check-prefix=DISABLED_MASKED_STRIDED 
; RUN: opt -mcpu=skx -S -loop-vectorize -instcombine -simplifycfg -force-vector-width=8 -force-vector-interleave=1 -enable-interleaved-mem-accesses  -enable-masked-interleaved-mem-accesses < %s | FileCheck %s -check-prefix=ENABLED_MASKED_STRIDED 

target datalayout = "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i386-unknown-linux-gnu"

; When masked-interleaved-groups are disabled:
; Check that the predicated load is not vectorized as an
; interleaved-group but rather as a scalarized accesses.
; (For SKX, Gather is not supported by the compiler for chars, therefore
;  the only remaining alternative is to scalarize).
; In this case a scalar epilogue is not needed.
;
; When  masked-interleave-group is enabled we expect to find the proper mask
; shuffling code, feeding the wide masked load for an interleave-group (with
; a single member).
; Since the last (second) member of the load-group is a gap, peeling is used,
; so we also expect to find a scalar epilogue loop.
;
; void masked_strided1(const unsigned char* restrict p,
;                      unsigned char* restrict q,
;                      unsigned char guard) {
;   for(ix=0; ix < 1024; ++ix) {
;     if (ix > guard) {
;         char t = p[2*ix];
;         q[ix] = t;
;     }
;   }
; }

;DISABLED_MASKED_STRIDED-LABEL: @masked_strided1(
;DISABLED_MASKED_STRIDED: vector.body:
;DISABLED_MASKED_STRIDED-NEXT:  %index = phi i32 
;DISABLED_MASKED_STRIDED-NEXT:  %[[VECIND:.+]] = phi <8 x i32> [ <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
;DISABLED_MASKED_STRIDED-NOT:   %interleaved.mask =
;DISABLED_MASKED_STRIDED-NOT:   call void @llvm.masked.load.
;DISABLED_MASKED_STRIDED-NOT:   %{{.*}} = shufflevector <16 x i8> %[[WIDEVEC]], <16 x i8> undef, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
;DISABLED_MASKED_STRIDED:       %[[VMASK:.+]] = icmp ugt <8 x i32> %[[VECIND]], %{{broadcast.splat*}}
;DISABLED_MASKED_STRIDED-NEXT:  %{{.*}} = shl nuw nsw <8 x i32> %[[VECIND]], <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
;DISABLED_MASKED_STRIDED-NEXT:  %[[M:.+]] = extractelement <8 x i1> %[[VMASK]], i32 0
;DISABLED_MASKED_STRIDED-NEXT:  br i1 %[[M]], label %pred.load.if, label %pred.load.continue
;DISABLED_MASKED_STRIDED-NOT:   %interleaved.mask =
;DISABLED_MASKED_STRIDED-NOT:   call void @llvm.masked.load.
;DISABLED_MASKED_STRIDED-NOT:   %{{.*}} = shufflevector <16 x i8> %{{.*}}, <16 x i8> undef, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
;DISABLED_MASKED_STRIDED-NOT: for.body:
;DISABLED_MASKED_STRIDED:     for.end:

;ENABLED_MASKED_STRIDED-LABEL: @masked_strided1(
;ENABLED_MASKED_STRIDED: vector.body:
;ENABLED_MASKED_STRIDED-NEXT:  %index = phi i32 
;ENABLED_MASKED_STRIDED-NEXT:  %[[VECIND:.+]] = phi <8 x i32> [ <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
;ENABLED_MASKED_STRIDED:       %[[VMASK:.+]] = icmp ugt <8 x i32> %[[VECIND]], %{{broadcast.splat*}}
;ENABLED_MASKED_STRIDED:       %interleaved.mask = shufflevector <8 x i1> %[[VMASK]], <8 x i1> undef, <16 x i32> <i32 0, i32 0, i32 1, i32 1, i32 2, i32 2, i32 3, i32 3, i32 4, i32 4, i32 5, i32 5, i32 6, i32 6, i32 7, i32 7>
;ENABLED_MASKED_STRIDED-NEXT:  %[[WIDEMASKEDLOAD:.+]] = call <16 x i8> @llvm.masked.load.v16i8.p0v16i8(<16 x i8>* %{{.*}}, i32 1, <16 x i1> %interleaved.mask, <16 x i8> undef)
;ENABLED_MASKED_STRIDED-NEXT:  %[[STRIDEDVEC:.+]] = shufflevector <16 x i8> %[[WIDEMASKEDLOAD]], <16 x i8> undef, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
;ENABLED_MASKED_STRIDED: for.body:

define dso_local void @masked_strided1(i8* noalias nocapture readonly %p, i8* noalias nocapture %q, i8 zeroext %guard) local_unnamed_addr {
entry:
  %conv = zext i8 %guard to i32
  br label %for.body

for.body:
  %ix.09 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp1 = icmp ugt i32 %ix.09, %conv
  br i1 %cmp1, label %if.then, label %for.inc

if.then:
  %mul = shl nuw nsw i32 %ix.09, 1
  %arrayidx = getelementptr inbounds i8, i8* %p, i32 %mul
  %0 = load i8, i8* %arrayidx, align 1
  %arrayidx3 = getelementptr inbounds i8, i8* %q, i32 %ix.09
  store i8 %0, i8* %arrayidx3, align 1
  br label %for.inc

for.inc:
  %inc = add nuw nsw i32 %ix.09, 1
  %exitcond = icmp eq i32 %inc, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

; Exactly the same scenario except we are now optimizing for size, therefore
; we check that no scalar epilogue is created. Since we can't create an epilog
; we need the ability to mask out the gaps.
; When enable-masked-interleaved-access is enabled, the interleave-groups will
; be vectorized with masked wide-loads with the mask properly shuffled and
; And-ed with the gaps mask.

;ENABLED_MASKED_STRIDED-LABEL: @masked_strided1_optsize(
;ENABLED_MASKED_STRIDED-NEXT:  entry:
;ENABLED_MASKED_STRIDED-NEXT:    [[CONV:%.*]] = zext i8 [[GUARD:%.*]] to i32
;ENABLED_MASKED_STRIDED-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <8 x i32> undef, i32 [[CONV]], i32 0
;ENABLED_MASKED_STRIDED-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <8 x i32> [[BROADCAST_SPLATINSERT]], <8 x i32> undef, <8 x i32> zeroinitializer
;ENABLED_MASKED_STRIDED-NEXT:    br label [[VECTOR_BODY:%.*]]
;ENABLED_MASKED_STRIDED:       vector.body:
;ENABLED_MASKED_STRIDED-NEXT:    [[INDEX:%.*]] = phi i32 [ 0, [[ENTRY:%.*]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
;ENABLED_MASKED_STRIDED-NEXT:    [[VEC_IND:%.*]] = phi <8 x i32> [ <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, [[ENTRY]] ], [ [[VEC_IND_NEXT:%.*]], [[VECTOR_BODY]] ]
;ENABLED_MASKED_STRIDED-NEXT:    [[TMP0:%.*]] = icmp ugt <8 x i32> [[VEC_IND]], [[BROADCAST_SPLAT]]
;ENABLED_MASKED_STRIDED-NEXT:    [[TMP1:%.*]] = shl nuw nsw i32 [[INDEX]], 1
;ENABLED_MASKED_STRIDED-NEXT:    [[TMP2:%.*]] = getelementptr inbounds i8, i8* [[P:%.*]], i32 [[TMP1]]
;ENABLED_MASKED_STRIDED-NEXT:    [[TMP3:%.*]] = bitcast i8* [[TMP2]] to <16 x i8>*
;ENABLED_MASKED_STRIDED-NEXT:    [[INTERLEAVED_MASK:%.*]] = shufflevector <8 x i1> [[TMP0]], <8 x i1> undef, <16 x i32> <i32 0, i32 0, i32 1, i32 1, i32 2, i32 2, i32 3, i32 3, i32 4, i32 4, i32 5, i32 5, i32 6, i32 6, i32 7, i32 7>
;ENABLED_MASKED_STRIDED-NEXT:    [[TMP4:%.*]] = and <16 x i1> [[INTERLEAVED_MASK]], <i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false>
;ENABLED_MASKED_STRIDED-NEXT:    [[WIDE_MASKED_VEC:%.*]] = call <16 x i8> @llvm.masked.load.v16i8.p0v16i8(<16 x i8>* [[TMP3]], i32 1, <16 x i1> [[TMP4]], <16 x i8> undef)
;ENABLED_MASKED_STRIDED-NEXT:    [[STRIDED_VEC:%.*]] = shufflevector <16 x i8> [[WIDE_MASKED_VEC]], <16 x i8> undef, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
;ENABLED_MASKED_STRIDED-NEXT:    [[TMP5:%.*]] = getelementptr inbounds i8, i8* [[Q:%.*]], i32 [[INDEX]]
;ENABLED_MASKED_STRIDED-NEXT:    [[TMP6:%.*]] = bitcast i8* [[TMP5]] to <8 x i8>*
;ENABLED_MASKED_STRIDED-NEXT:    call void @llvm.masked.store.v8i8.p0v8i8(<8 x i8> [[STRIDED_VEC]], <8 x i8>* [[TMP6]], i32 1, <8 x i1> [[TMP0]])
;ENABLED_MASKED_STRIDED-NEXT:    [[INDEX_NEXT]] = add i32 [[INDEX]], 8
;ENABLED_MASKED_STRIDED-NEXT:    [[VEC_IND_NEXT]] = add <8 x i32> [[VEC_IND]], <i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8>
;ENABLED_MASKED_STRIDED-NEXT:    [[TMP7:%.*]] = icmp eq i32 [[INDEX_NEXT]], 1024
;ENABLED_MASKED_STRIDED-NEXT:    br i1 [[TMP7]]
;ENABLED_MASKED_STRIDED-NOT:   for.body:
;ENABLED_MASKED_STRIDED:       for.end:
;ENABLED_MASKED_STRIDED-NEXT:    ret void


define dso_local void @masked_strided1_optsize(i8* noalias nocapture readonly %p, i8* noalias nocapture %q, i8 zeroext %guard) local_unnamed_addr optsize {
entry:
  %conv = zext i8 %guard to i32
  br label %for.body

for.body:
  %ix.09 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp1 = icmp ugt i32 %ix.09, %conv
  br i1 %cmp1, label %if.then, label %for.inc

if.then:
  %mul = shl nuw nsw i32 %ix.09, 1
  %arrayidx = getelementptr inbounds i8, i8* %p, i32 %mul
  %0 = load i8, i8* %arrayidx, align 1
  %arrayidx3 = getelementptr inbounds i8, i8* %q, i32 %ix.09
  store i8 %0, i8* %arrayidx3, align 1
  br label %for.inc

for.inc:
  %inc = add nuw nsw i32 %ix.09, 1
  %exitcond = icmp eq i32 %inc, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}


; Accesses with gaps under Optsize scenario again, with unknown trip-count
; this time, in order to check the behavior of folding-the-tail (folding the
; remainder loop into the main loop using masking) together with interleaved-
; groups.
; When masked-interleave-group is disabled the interleave-groups will be
; invalidated during Legality checks; So there we check for no epilogue
; and for scalarized conditional accesses.
; When masked-interleave-group is enabled we check that there is no epilogue,
; and that the interleave-groups are vectorized using proper masking (with
; shuffling of the mask feeding the wide masked load/store).
; The mask itself is an And of two masks: one that masks away the remainder
; iterations, and one that masks away the 'else' of the 'if' statement.
; The shuffled mask is also And-ed with the gaps mask.
;
; void masked_strided1_optsize_unknown_tc(const unsigned char* restrict p,
;                      unsigned char* restrict q,
;                      unsigned char guard,
;                      int n) {
;   for(ix=0; ix < n; ++ix) {
;     if (ix > guard) {
;         char t = p[2*ix];
;         q[ix] = t;
;     }
;   }
; }

; DISABLED_MASKED_STRIDED-LABEL: @masked_strided1_optsize_unknown_tc(
; DISABLED_MASKED_STRIDED:       vector.body:
; DISABLED_MASKED_STRIDED-NEXT:    [[INDEX:%.*]] = phi i32 
; DISABLED_MASKED_STRIDED-NEXT:    [[VEC_IND:%.*]] = phi <8 x i32> [ <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; DISABLED_MASKED_STRIDED-NEXT:    [[TMP0:%.*]] = icmp ugt <8 x i32> [[VEC_IND]], {{.*}}
; DISABLED_MASKED_STRIDED-NEXT:    [[TMP1:%.*]] = shl nuw nsw <8 x i32> [[VEC_IND]], <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
; DISABLED_MASKED_STRIDED-NEXT:    [[TMP2:%.*]] = icmp ule <8 x i32> [[VEC_IND]], {{.*}}
; DISABLED_MASKED_STRIDED-NEXT:    [[TMP3:%.*]] = and <8 x i1> [[TMP0]], [[TMP2]]
; DISABLED_MASKED_STRIDED-NEXT:    [[TMP4:%.*]] = extractelement <8 x i1> [[TMP3]], i32 0
; DISABLED_MASKED_STRIDED-NEXT:    br i1 [[TMP4]], label [[PRED_LOAD_IF:%.*]], label [[PRED_LOAD_CONTINUE:%.*]]
; DISABLED_MASKED_STRIDED:       pred.load.if:
; DISABLED_MASKED_STRIDED-NEXT:    [[TMP5:%.*]] = extractelement <8 x i32> [[TMP1]], i32 0
; DISABLED_MASKED_STRIDED-NEXT:    [[TMP6:%.*]] = getelementptr inbounds i8, i8* [[P:%.*]], i32 [[TMP5]]
; DISABLED_MASKED_STRIDED-NEXT:    [[TMP7:%.*]] = load i8, i8* [[TMP6]], align 1
; DISABLED_MASKED_STRIDED-NEXT:    [[TMP8:%.*]] = insertelement <8 x i8> undef, i8 [[TMP7]], i32 0
; DISABLED_MASKED_STRIDED-NEXT:    br label [[PRED_LOAD_CONTINUE]]
; DISABLED_MASKED_STRIDED-NOT:   for.body:
; DISABLED_MASKED_STRIDED:       for.end:
; DISABLED_MASKED_STRIDED-NEXT:    ret void


; ENABLED_MASKED_STRIDED-LABEL: @masked_strided1_optsize_unknown_tc(
; ENABLED_MASKED_STRIDED-NEXT:  entry:
; ENABLED_MASKED_STRIDED-NEXT:    [[CMP9:%.*]] = icmp sgt i32 [[N:%.*]], 0
; ENABLED_MASKED_STRIDED-NEXT:    br i1 [[CMP9]], label [[VECTOR_PH:%.*]], label [[FOR_END:%.*]]
; ENABLED_MASKED_STRIDED:       vector.ph:
; ENABLED_MASKED_STRIDED-NEXT:    [[CONV:%.*]] = zext i8 [[GUARD:%.*]] to i32
; ENABLED_MASKED_STRIDED-NEXT:    [[N_RND_UP:%.*]] = add i32 [[N]], 7
; ENABLED_MASKED_STRIDED-NEXT:    [[N_VEC:%.*]] = and i32 [[N_RND_UP]], -8
; ENABLED_MASKED_STRIDED-NEXT:    [[TRIP_COUNT_MINUS_1:%.*]] = add i32 [[N]], -1
; ENABLED_MASKED_STRIDED-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <8 x i32> undef, i32 [[CONV]], i32 0
; ENABLED_MASKED_STRIDED-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <8 x i32> [[BROADCAST_SPLATINSERT]], <8 x i32> undef, <8 x i32> zeroinitializer
; ENABLED_MASKED_STRIDED-NEXT:    [[BROADCAST_SPLATINSERT1:%.*]] = insertelement <8 x i32> undef, i32 [[TRIP_COUNT_MINUS_1]], i32 0
; ENABLED_MASKED_STRIDED-NEXT:    [[BROADCAST_SPLAT2:%.*]] = shufflevector <8 x i32> [[BROADCAST_SPLATINSERT1]], <8 x i32> undef, <8 x i32> zeroinitializer
; ENABLED_MASKED_STRIDED-NEXT:    br label [[VECTOR_BODY:%.*]]
; ENABLED_MASKED_STRIDED:       vector.body:
; ENABLED_MASKED_STRIDED-NEXT:    [[INDEX:%.*]] = phi i32 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; ENABLED_MASKED_STRIDED-NEXT:    [[VEC_IND:%.*]] = phi <8 x i32> [ <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, [[VECTOR_PH]] ], [ [[VEC_IND_NEXT:%.*]], [[VECTOR_BODY]] ]
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP0:%.*]] = icmp ugt <8 x i32> [[VEC_IND]], [[BROADCAST_SPLAT]]
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP1:%.*]] = shl nuw nsw i32 [[INDEX]], 1
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP2:%.*]] = getelementptr inbounds i8, i8* [[P:%.*]], i32 [[TMP1]]
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP3:%.*]] = icmp ule <8 x i32> [[VEC_IND]], [[BROADCAST_SPLAT2]]
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP4:%.*]] = and <8 x i1> [[TMP0]], [[TMP3]]
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP5:%.*]] = bitcast i8* [[TMP2]] to <16 x i8>*
; ENABLED_MASKED_STRIDED-NEXT:    [[INTERLEAVED_MASK:%.*]] = shufflevector <8 x i1> [[TMP4]], <8 x i1> undef, <16 x i32> <i32 0, i32 0, i32 1, i32 1, i32 2, i32 2, i32 3, i32 3, i32 4, i32 4, i32 5, i32 5, i32 6, i32 6, i32 7, i32 7>
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP6:%.*]] = and <16 x i1> [[INTERLEAVED_MASK]], <i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false>
; ENABLED_MASKED_STRIDED-NEXT:    [[WIDE_MASKED_VEC:%.*]] = call <16 x i8> @llvm.masked.load.v16i8.p0v16i8(<16 x i8>* [[TMP5]], i32 1, <16 x i1> [[TMP6]], <16 x i8> undef)
; ENABLED_MASKED_STRIDED-NEXT:    [[STRIDED_VEC:%.*]] = shufflevector <16 x i8> [[WIDE_MASKED_VEC]], <16 x i8> undef, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP7:%.*]] = getelementptr inbounds i8, i8* [[Q:%.*]], i32 [[INDEX]]
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP8:%.*]] = bitcast i8* [[TMP7]] to <8 x i8>*
; ENABLED_MASKED_STRIDED-NEXT:    call void @llvm.masked.store.v8i8.p0v8i8(<8 x i8> [[STRIDED_VEC]], <8 x i8>* [[TMP8]], i32 1, <8 x i1> [[TMP4]])
; ENABLED_MASKED_STRIDED-NEXT:    [[INDEX_NEXT]] = add i32 [[INDEX]], 8
; ENABLED_MASKED_STRIDED-NEXT:    [[VEC_IND_NEXT]] = add <8 x i32> [[VEC_IND]], <i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8>
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP9:%.*]] = icmp eq i32 [[INDEX_NEXT]], [[N_VEC]]
; ENABLED_MASKED_STRIDED-NEXT:    br i1 [[TMP9]], label [[FOR_END]], label [[VECTOR_BODY]]
; ENABLED_MASKED_STRIDED-NOT:   for.body:
; ENABLED_MASKED_STRIDED:       for.end:
; ENABLED_MASKED_STRIDED-NEXT:    ret void

define dso_local void @masked_strided1_optsize_unknown_tc(i8* noalias nocapture readonly %p, i8* noalias nocapture %q, i8 zeroext %guard, i32 %n) local_unnamed_addr optsize {
entry:
  %cmp9 = icmp sgt i32 %n, 0
  br i1 %cmp9, label %for.body.lr.ph, label %for.end

for.body.lr.ph:
  %conv = zext i8 %guard to i32
  br label %for.body

for.body:
  %ix.010 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.inc ]
  %cmp1 = icmp ugt i32 %ix.010, %conv
  br i1 %cmp1, label %if.then, label %for.inc

if.then:
  %mul = shl nuw nsw i32 %ix.010, 1
  %arrayidx = getelementptr inbounds i8, i8* %p, i32 %mul
  %0 = load i8, i8* %arrayidx, align 1
  %arrayidx3 = getelementptr inbounds i8, i8* %q, i32 %ix.010
  store i8 %0, i8* %arrayidx3, align 1
  br label %for.inc

for.inc:
  %inc = add nuw nsw i32 %ix.010, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}

; Same, with stride 3. This is to check the gaps-mask and the shuffled mask
; with a different stride.
; So accesses are with gaps under Optsize scenario again, with unknown trip-
; count, in order to check the behavior of folding-the-tail (folding the
; remainder loop into the main loop using masking) together with interleaved-
; groups.
; When masked-interleave-group is enabled we check that there is no epilogue,
; and that the interleave-groups are vectorized using proper masking (with
; shuffling of the mask feeding the wide masked load/store).
; The mask itself is an And of two masks: one that masks away the remainder
; iterations, and one that masks away the 'else' of the 'if' statement.
; The shuffled mask is also And-ed with the gaps mask.
;
; void masked_strided3_optsize_unknown_tc(const unsigned char* restrict p,
;                      unsigned char* restrict q,
;                      unsigned char guard,
;                      int n) {
;   for(ix=0; ix < n; ++ix) {
;     if (ix > guard) {
;         char t = p[3*ix];
;         q[ix] = t;
;     }
;   }
; }


; ENABLED_MASKED_STRIDED-LABEL: @masked_strided3_optsize_unknown_tc(
; ENABLED_MASKED_STRIDED-NEXT:  entry:
; ENABLED_MASKED_STRIDED-NEXT:    [[CMP9:%.*]] = icmp sgt i32 [[N:%.*]], 0
; ENABLED_MASKED_STRIDED-NEXT:    br i1 [[CMP9]], label [[VECTOR_PH:%.*]], label [[FOR_END:%.*]]
; ENABLED_MASKED_STRIDED:       vector.ph:
; ENABLED_MASKED_STRIDED-NEXT:    [[CONV:%.*]] = zext i8 [[GUARD:%.*]] to i32
; ENABLED_MASKED_STRIDED-NEXT:    [[N_RND_UP:%.*]] = add i32 [[N]], 7
; ENABLED_MASKED_STRIDED-NEXT:    [[N_VEC:%.*]] = and i32 [[N_RND_UP]], -8
; ENABLED_MASKED_STRIDED-NEXT:    [[TRIP_COUNT_MINUS_1:%.*]] = add i32 [[N]], -1
; ENABLED_MASKED_STRIDED-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <8 x i32> undef, i32 [[CONV]], i32 0
; ENABLED_MASKED_STRIDED-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <8 x i32> [[BROADCAST_SPLATINSERT]], <8 x i32> undef, <8 x i32> zeroinitializer
; ENABLED_MASKED_STRIDED-NEXT:    [[BROADCAST_SPLATINSERT1:%.*]] = insertelement <8 x i32> undef, i32 [[TRIP_COUNT_MINUS_1]], i32 0
; ENABLED_MASKED_STRIDED-NEXT:    [[BROADCAST_SPLAT2:%.*]] = shufflevector <8 x i32> [[BROADCAST_SPLATINSERT1]], <8 x i32> undef, <8 x i32> zeroinitializer
; ENABLED_MASKED_STRIDED-NEXT:    br label [[VECTOR_BODY:%.*]]
; ENABLED_MASKED_STRIDED:       vector.body:
; ENABLED_MASKED_STRIDED-NEXT:    [[INDEX:%.*]] = phi i32 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; ENABLED_MASKED_STRIDED-NEXT:    [[VEC_IND:%.*]] = phi <8 x i32> [ <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, [[VECTOR_PH]] ], [ [[VEC_IND_NEXT:%.*]], [[VECTOR_BODY]] ]
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP0:%.*]] = icmp ugt <8 x i32> [[VEC_IND]], [[BROADCAST_SPLAT]]
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP1:%.*]] = mul nsw i32 [[INDEX]], 3
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP2:%.*]] = getelementptr inbounds i8, i8* [[P:%.*]], i32 [[TMP1]]
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP3:%.*]] = icmp ule <8 x i32> [[VEC_IND]], [[BROADCAST_SPLAT2]]
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP4:%.*]] = and <8 x i1> [[TMP0]], [[TMP3]]
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP5:%.*]] = bitcast i8* [[TMP2]] to <24 x i8>*
; ENABLED_MASKED_STRIDED-NEXT:    [[INTERLEAVED_MASK:%.*]] = shufflevector <8 x i1> [[TMP4]], <8 x i1> undef, <24 x i32> <i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 2, i32 2, i32 2, i32 3, i32 3, i32 3, i32 4, i32 4, i32 4, i32 5, i32 5, i32 5, i32 6, i32 6, i32 6, i32 7, i32 7, i32 7>
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP6:%.*]] = and <24 x i1> [[INTERLEAVED_MASK]], <i1 true, i1 false, i1 false, i1 true, i1 false, i1 false, i1 true, i1 false, i1 false, i1 true, i1 false, i1 false, i1 true, i1 false, i1 false, i1 true, i1 false, i1 false, i1 true, i1 false, i1 false, i1 true, i1 false, i1 false>
; ENABLED_MASKED_STRIDED-NEXT:    [[WIDE_MASKED_VEC:%.*]] = call <24 x i8> @llvm.masked.load.v24i8.p0v24i8(<24 x i8>* [[TMP5]], i32 1, <24 x i1> [[TMP6]], <24 x i8> undef)
; ENABLED_MASKED_STRIDED-NEXT:    [[STRIDED_VEC:%.*]] = shufflevector <24 x i8> [[WIDE_MASKED_VEC]], <24 x i8> undef, <8 x i32> <i32 0, i32 3, i32 6, i32 9, i32 12, i32 15, i32 18, i32 21>
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP7:%.*]] = getelementptr inbounds i8, i8* [[Q:%.*]], i32 [[INDEX]]
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP8:%.*]] = bitcast i8* [[TMP7]] to <8 x i8>*
; ENABLED_MASKED_STRIDED-NEXT:    call void @llvm.masked.store.v8i8.p0v8i8(<8 x i8> [[STRIDED_VEC]], <8 x i8>* [[TMP8]], i32 1, <8 x i1> [[TMP4]])
; ENABLED_MASKED_STRIDED-NEXT:    [[INDEX_NEXT]] = add i32 [[INDEX]], 8
; ENABLED_MASKED_STRIDED-NEXT:    [[VEC_IND_NEXT]] = add <8 x i32> [[VEC_IND]], <i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8>
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP9:%.*]] = icmp eq i32 [[INDEX_NEXT]], [[N_VEC]]
; ENABLED_MASKED_STRIDED-NEXT:    br i1 [[TMP9]], label [[FOR_END]], label [[VECTOR_BODY]]
; ENABLED_MASKED_STRIDED:       for.end:
; ENABLED_MASKED_STRIDED-NEXT:    ret void
;
define dso_local void @masked_strided3_optsize_unknown_tc(i8* noalias nocapture readonly %p, i8* noalias nocapture %q, i8 zeroext %guard, i32 %n) local_unnamed_addr optsize {
entry:
  %cmp9 = icmp sgt i32 %n, 0
  br i1 %cmp9, label %for.body.lr.ph, label %for.end

for.body.lr.ph:
  %conv = zext i8 %guard to i32
  br label %for.body

for.body:
  %ix.010 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.inc ]
  %cmp1 = icmp ugt i32 %ix.010, %conv
  br i1 %cmp1, label %if.then, label %for.inc

if.then:
  %mul = mul nsw i32 %ix.010, 3
  %arrayidx = getelementptr inbounds i8, i8* %p, i32 %mul
  %0 = load i8, i8* %arrayidx, align 1
  %arrayidx3 = getelementptr inbounds i8, i8* %q, i32 %ix.010
  store i8 %0, i8* %arrayidx3, align 1
  br label %for.inc

for.inc:
  %inc = add nuw nsw i32 %ix.010, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}


; Back to stride 2 with gaps with a known trip count under opt for size,
; but this time the load/store are not predicated. 
; When enable-masked-interleaved-access is disabled, the interleave-groups will
; be invalidated during cost-model checks because we have gaps and we can't
; create an epilog. The access is thus scalarized.
; (Before the fix that this test checks, we used to create an epilogue despite
; optsize, and vectorized the access as an interleaved-group. This is now fixed,
; and we make sure that a scalar epilogue does not exist).
; When enable-masked-interleaved-access is enabled, the interleave-groups will
; be vectorized with masked wide-loads (masking away the gaps).
;
; void unconditional_strided1_optsize(const unsigned char* restrict p,
;                                unsigned char* restrict q,
;                                unsigned char guard) {
;   for(ix=0; ix < 1024; ++ix) {
;         char t = p[2*ix];
;         q[ix] = t;
;   }
; }

;DISABLED_MASKED_STRIDED-LABEL: @unconditional_strided1_optsize(
;DISABLED_MASKED_STRIDED: vector.body:
;DISABLED_MASKED_STRIDED-NOT: call <16 x i8> @llvm.masked.load.v16i8.p0v16i8
;DISABLED_MASKED_STRIDED:     %{{.*}} = extractelement <8 x i32> %{{.*}}, i32 0       
;DISABLED_MASKED_STRIDED-NOT: for.body:
;DISABLED_MASKED_STRIDED:     for.end:

;ENABLED_MASKED_STRIDED-LABEL: @unconditional_strided1_optsize(
;ENABLED_MASKED_STRIDED-NEXT:  entry:
;ENABLED_MASKED_STRIDED-NEXT:    br label [[VECTOR_BODY:%.*]]
;ENABLED_MASKED_STRIDED:       vector.body:
;ENABLED_MASKED_STRIDED-NEXT:    [[INDEX:%.*]] = phi i32 [ 0, [[ENTRY:%.*]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
;ENABLED_MASKED_STRIDED-NEXT:    [[TMP0:%.*]] = shl nuw nsw i32 [[INDEX]], 1
;ENABLED_MASKED_STRIDED-NEXT:    [[TMP1:%.*]] = getelementptr inbounds i8, i8* [[P:%.*]], i32 [[TMP0]]
;ENABLED_MASKED_STRIDED-NEXT:    [[TMP2:%.*]] = bitcast i8* [[TMP1]] to <16 x i8>*
;ENABLED_MASKED_STRIDED-NEXT:    [[WIDE_MASKED_VEC:%.*]] = call <16 x i8> @llvm.masked.load.v16i8.p0v16i8(<16 x i8>* [[TMP2]], i32 1, <16 x i1> <i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false>, <16 x i8> undef)
;ENABLED_MASKED_STRIDED-NEXT:    [[STRIDED_VEC:%.*]] = shufflevector <16 x i8> [[WIDE_MASKED_VEC]], <16 x i8> undef, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
;ENABLED_MASKED_STRIDED-NEXT:    [[TMP3:%.*]] = getelementptr inbounds i8, i8* [[Q:%.*]], i32 [[INDEX]]
;ENABLED_MASKED_STRIDED-NEXT:    [[TMP4:%.*]] = bitcast i8* [[TMP3]] to <8 x i8>*
;ENABLED_MASKED_STRIDED-NEXT:    store <8 x i8> [[STRIDED_VEC]], <8 x i8>* [[TMP4]], align 1
;ENABLED_MASKED_STRIDED-NEXT:    [[INDEX_NEXT]] = add i32 [[INDEX]], 8
;ENABLED_MASKED_STRIDED-NEXT:    [[TMP5:%.*]] = icmp eq i32 [[INDEX_NEXT]], 1024
;ENABLED_MASKED_STRIDED-NEXT:    br i1 [[TMP5]], label [[FOR_END:%.*]], label [[VECTOR_BODY]]
;ENABLED_MASKED_STRIDED-NOT:   for.body:
;ENABLED_MASKED_STRIDED:       for.end:
;ENABLED_MASKED_STRIDED-NEXT:    ret void


define dso_local void @unconditional_strided1_optsize(i8* noalias nocapture readonly %p, i8* noalias nocapture %q, i8 zeroext %guard) local_unnamed_addr optsize {
entry:
  br label %for.body

for.body:
  %ix.06 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %mul = shl nuw nsw i32 %ix.06, 1
  %arrayidx = getelementptr inbounds i8, i8* %p, i32 %mul
  %0 = load i8, i8* %arrayidx, align 1
  %arrayidx1 = getelementptr inbounds i8, i8* %q, i32 %ix.06
  store i8 %0, i8* %arrayidx1, align 1
  %inc = add nuw nsw i32 %ix.06, 1
  %exitcond = icmp eq i32 %inc, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}



; Unconditioal accesses with gaps under Optsize scenario again, with unknown
; trip-count this time, in order to check the behavior of folding-the-tail 
; (folding the remainder loop into the main loop using masking) together with
; interleaved-groups. Folding-the-tail turns the accesses to conditional which
; requires proper masking. In addition we need to mask out the gaps (all
; because we are not allowed to use an epilog due to optsize).
; When enable-masked-interleaved-access is disabled, the interleave-groups will
; be invalidated during cost-model checks. So there we check for no epilogue
; and for scalarized conditional accesses.
; When masked-interleave-group is enabled we check that there is no epilogue,
; and that the interleave-groups are vectorized using proper masking (with
; shuffling of the mask feeding the wide masked load/store).
; The shuffled mask is also And-ed with the gaps mask.
;
;   for(ix=0; ix < n; ++ix) {
;         char t = p[2*ix];
;         q[ix] = t;
;   }

; DISABLED_MASKED_STRIDED-LABEL: @unconditional_strided1_optsize_unknown_tc(
; DISABLED_MASKED_STRIDED:       vector.body:
; DISABLED_MASKED_STRIDED-NEXT:    [[INDEX:%.*]] = phi i32 
; DISABLED_MASKED_STRIDED-NEXT:    [[VEC_IND:%.*]] = phi <8 x i32> [ <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; DISABLED_MASKED_STRIDED-NEXT:    [[TMP0:%.*]] = shl nuw nsw <8 x i32> [[VEC_IND]], <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
; DISABLED_MASKED_STRIDED-NEXT:    [[TMP1:%.*]] = icmp ule <8 x i32> [[VEC_IND]], {{.*}} 
; DISABLED_MASKED_STRIDED-NEXT:    [[TMP2:%.*]] = extractelement <8 x i1> [[TMP1]], i32 0
; DISABLED_MASKED_STRIDED-NEXT:    br i1 [[TMP2]], label [[PRED_LOAD_IF:%.*]], label [[PRED_LOAD_CONTINUE:%.*]]
; DISABLED_MASKED_STRIDED:       pred.load.if:
; DISABLED_MASKED_STRIDED-NEXT:    [[TMP3:%.*]] = extractelement <8 x i32> [[TMP0]], i32 0
; DISABLED_MASKED_STRIDED-NEXT:    [[TMP4:%.*]] = getelementptr inbounds i8, i8* [[P:%.*]], i32 [[TMP3]]
; DISABLED_MASKED_STRIDED-NEXT:    [[TMP5:%.*]] = load i8, i8* [[TMP4]], align 1
; DISABLED_MASKED_STRIDED-NEXT:    [[TMP6:%.*]] = insertelement <8 x i8> undef, i8 [[TMP5]], i32 0
; DISABLED_MASKED_STRIDED-NEXT:    br label [[PRED_LOAD_CONTINUE]]
; DISBLED_MASKED_STRIDED-NOT:    for.body:
; DISABLED_MASKED_STRIDED:       for.end:
; DISABLED_MASKED_STRIDED-NEXT:    ret void

; ENABLED_MASKED_STRIDED-LABEL: @unconditional_strided1_optsize_unknown_tc(
; ENABLED_MASKED_STRIDED-NEXT:  entry:
; ENABLED_MASKED_STRIDED-NEXT:    [[CMP6:%.*]] = icmp sgt i32 [[N:%.*]], 0
; ENABLED_MASKED_STRIDED-NEXT:    br i1 [[CMP6]], label [[VECTOR_PH:%.*]], label [[FOR_END:%.*]]
; ENABLED_MASKED_STRIDED:       vector.ph:
; ENABLED_MASKED_STRIDED-NEXT:    [[N_RND_UP:%.*]] = add i32 [[N]], 7
; ENABLED_MASKED_STRIDED-NEXT:    [[N_VEC:%.*]] = and i32 [[N_RND_UP]], -8
; ENABLED_MASKED_STRIDED-NEXT:    [[TRIP_COUNT_MINUS_1:%.*]] = add i32 [[N]], -1
; ENABLED_MASKED_STRIDED-NEXT:    [[BROADCAST_SPLATINSERT1:%.*]] = insertelement <8 x i32> undef, i32 [[TRIP_COUNT_MINUS_1]], i32 0
; ENABLED_MASKED_STRIDED-NEXT:    [[BROADCAST_SPLAT2:%.*]] = shufflevector <8 x i32> [[BROADCAST_SPLATINSERT1]], <8 x i32> undef, <8 x i32> zeroinitializer
; ENABLED_MASKED_STRIDED-NEXT:    br label [[VECTOR_BODY:%.*]]
; ENABLED_MASKED_STRIDED:       vector.body:
; ENABLED_MASKED_STRIDED-NEXT:    [[INDEX:%.*]] = phi i32 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; ENABLED_MASKED_STRIDED-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <8 x i32> undef, i32 [[INDEX]], i32 0
; ENABLED_MASKED_STRIDED-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <8 x i32> [[BROADCAST_SPLATINSERT]], <8 x i32> undef, <8 x i32> zeroinitializer
; ENABLED_MASKED_STRIDED-NEXT:    [[INDUCTION:%.*]] = add <8 x i32> [[BROADCAST_SPLAT]], <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP0:%.*]] = shl nuw nsw i32 [[INDEX]], 1
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP1:%.*]] = getelementptr inbounds i8, i8* [[P:%.*]], i32 [[TMP0]]
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP2:%.*]] = icmp ule <8 x i32> [[INDUCTION]], [[BROADCAST_SPLAT2]]
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP3:%.*]] = bitcast i8* [[TMP1]] to <16 x i8>*
; ENABLED_MASKED_STRIDED-NEXT:    [[INTERLEAVED_MASK:%.*]] = shufflevector <8 x i1> [[TMP2]], <8 x i1> undef, <16 x i32> <i32 0, i32 0, i32 1, i32 1, i32 2, i32 2, i32 3, i32 3, i32 4, i32 4, i32 5, i32 5, i32 6, i32 6, i32 7, i32 7>
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP4:%.*]] = and <16 x i1> [[INTERLEAVED_MASK]], <i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false>
; ENABLED_MASKED_STRIDED-NEXT:    [[WIDE_MASKED_VEC:%.*]] = call <16 x i8> @llvm.masked.load.v16i8.p0v16i8(<16 x i8>* [[TMP3]], i32 1, <16 x i1> [[TMP4]], <16 x i8> undef)
; ENABLED_MASKED_STRIDED-NEXT:    [[STRIDED_VEC:%.*]] = shufflevector <16 x i8> [[WIDE_MASKED_VEC]], <16 x i8> undef, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP5:%.*]] = getelementptr inbounds i8, i8* [[Q:%.*]], i32 [[INDEX]]
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP6:%.*]] = bitcast i8* [[TMP5]] to <8 x i8>*
; ENABLED_MASKED_STRIDED-NEXT:    call void @llvm.masked.store.v8i8.p0v8i8(<8 x i8> [[STRIDED_VEC]], <8 x i8>* [[TMP6]], i32 1, <8 x i1> [[TMP2]])
; ENABLED_MASKED_STRIDED-NEXT:    [[INDEX_NEXT]] = add i32 [[INDEX]], 8
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP7:%.*]] = icmp eq i32 [[INDEX_NEXT]], [[N_VEC]]
; ENABLED_MASKED_STRIDED-NEXT:    br i1 [[TMP7]], label [[FOR_END]], label [[VECTOR_BODY]]
; ENABLED_MASKED_STRIDED-NOT:   for.body:
; ENABLED_MASKED_STRIDED:       for.end:
; ENABLED_MASKED_STRIDED-NEXT:    ret void

define dso_local void @unconditional_strided1_optsize_unknown_tc(i8* noalias nocapture readonly %p, i8* noalias nocapture %q, i32 %n) local_unnamed_addr optsize {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body.preheader, label %for.end

for.body.preheader:
  br label %for.body

for.body:
  %ix.07 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %mul = shl nuw nsw i32 %ix.07, 1
  %arrayidx = getelementptr inbounds i8, i8* %p, i32 %mul
  %0 = load i8, i8* %arrayidx, align 1
  %arrayidx1 = getelementptr inbounds i8, i8* %q, i32 %ix.07
  store i8 %0, i8* %arrayidx1, align 1
  %inc = add nuw nsw i32 %ix.07, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}


; Check also a scenario with full interleave-groups (no gaps) as well as both
; load and store groups. We check that when masked-interleave-group is disabled
; the predicated loads (and stores) are not vectorized as an
; interleaved-group but rather as four separate scalarized accesses.
; (For SKX, gather/scatter is not supported by the compiler for chars, therefore
; the only remaining alternative is to scalarize).
; When  masked-interleave-group is enabled we expect to find the proper mask
; shuffling code, feeding the wide masked load/store for the two interleave-
; groups.
;
; void masked_strided2(const unsigned char* restrict p,
;                     unsigned char* restrict q,
;                     unsigned char guard) {
; for(ix=0; ix < 1024; ++ix) {
;     if (ix > guard) {
;         char left = p[2*ix];
;         char right = p[2*ix + 1];
;         char max = max(left, right);
;         q[2*ix] = max;
;         q[2*ix+1] = 0 - max;
;     }
; }
;}

;DISABLED_MASKED_STRIDED-LABEL: @masked_strided2(
;DISABLED_MASKED_STRIDED: vector.body:
;DISABLED_MASKED_STRIDED-NEXT:  %index = phi i32 
;DISABLED_MASKED_STRIDED-NEXT:  %[[VECIND:.+]] = phi <8 x i32> [ <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
;DISABLED_MASKED_STRIDED-NOT:   %interleaved.mask =
;DISABLED_MASKED_STRIDED-NOT:   call void @llvm.masked.load.
;DISABLED_MASKED_STRIDED-NOT:   call void @llvm.masked.store.
;DISABLED_MASKED_STRIDED-NOT:   %{{.*}} = shufflevector <16 x i8> %{{.*}}, <16 x i8> undef, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
;DISABLED_MASKED_STRIDED:        %[[VMASK:.+]] = icmp ugt <8 x i32> %[[VECIND]], %{{broadcast.splat*}}
;DISABLED_MASKED_STRIDED-NEXT:  %{{.*}} = shl nuw nsw <8 x i32> %[[VECIND]], <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
;DISABLED_MASKED_STRIDED-NEXT:  %[[M:.+]] = extractelement <8 x i1> %[[VMASK]], i32 0
;DISABLED_MASKED_STRIDED-NEXT:  br i1 %[[M]], label %pred.load.if, label %pred.load.continue
;DISABLED_MASKED_STRIDED-NOT:   %interleaved.mask =
;DISABLED_MASKED_STRIDED-NOT:   call void @llvm.masked.load.
;DISABLED_MASKED_STRIDED-NOT:   call void @llvm.masked.store.
;DISABLED_MASKED_STRIDED-NOT:   %{{.*}} = shufflevector <16 x i8> %{{.*}}, <16 x i8> undef, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>

;ENABLED_MASKED_STRIDED-LABEL: @masked_strided2(
;ENABLED_MASKED_STRIDED: vector.body:
;ENABLED_MASKED_STRIDED-NEXT:  %index = phi i32
;ENABLED_MASKED_STRIDED-NEXT:  %[[VECIND:.+]] = phi <8 x i32> [ <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
;ENABLED_MASKED_STRIDED:       %[[VMASK:.+]] = icmp ugt <8 x i32> %[[VECIND]], %{{broadcast.splat*}}
;ENABLED_MASKED_STRIDED:       %interleaved.mask = shufflevector <8 x i1> %[[VMASK]], <8 x i1> undef, <16 x i32> <i32 0, i32 0, i32 1, i32 1, i32 2, i32 2, i32 3, i32 3, i32 4, i32 4, i32 5, i32 5, i32 6, i32 6, i32 7, i32 7>
;ENABLED_MASKED_STRIDED-NEXT:  %[[WIDEMASKEDLOAD:.+]] = call <16 x i8> @llvm.masked.load.v16i8.p0v16i8(<16 x i8>* %{{.*}}, i32 1, <16 x i1> %interleaved.mask, <16 x i8> undef)
;ENABLED_MASKED_STRIDED-NEXT:  %{{.*}} = shufflevector <16 x i8> %[[WIDEMASKEDLOAD]], <16 x i8> undef, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
;ENABLED_MASKED_STRIDED-NEXT:  %{{.*}} = shufflevector <16 x i8> %[[WIDEMASKEDLOAD]], <16 x i8> undef, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
;ENABLED_MASKED_STRIDED:       call void @llvm.masked.store.v16i8.p0v16i8(<16 x i8> %{{.*}}, <16 x i8>* %{{.*}}, i32 1, <16 x i1> %interleaved.mask)

; Function Attrs: norecurse nounwind
define dso_local void @masked_strided2(i8* noalias nocapture readonly %p, i8* noalias nocapture %q, i8 zeroext %guard) local_unnamed_addr  {
entry:
  %conv = zext i8 %guard to i32
  br label %for.body

for.body:
  %ix.024 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp1 = icmp ugt i32 %ix.024, %conv
  br i1 %cmp1, label %if.then, label %for.inc

if.then:
  %mul = shl nuw nsw i32 %ix.024, 1
  %arrayidx = getelementptr inbounds i8, i8* %p, i32 %mul
  %0 = load i8, i8* %arrayidx, align 1
  %add = or i32 %mul, 1
  %arrayidx4 = getelementptr inbounds i8, i8* %p, i32 %add
  %1 = load i8, i8* %arrayidx4, align 1
  %cmp.i = icmp slt i8 %0, %1
  %spec.select.i = select i1 %cmp.i, i8 %1, i8 %0
  %arrayidx6 = getelementptr inbounds i8, i8* %q, i32 %mul
  store i8 %spec.select.i, i8* %arrayidx6, align 1
  %sub = sub i8 0, %spec.select.i
  %arrayidx11 = getelementptr inbounds i8, i8* %q, i32 %add
  store i8 %sub, i8* %arrayidx11, align 1
  br label %for.inc

for.inc:
  %inc = add nuw nsw i32 %ix.024, 1
  %exitcond = icmp eq i32 %inc, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

; Full groups again, this time checking an Optsize scenario, with unknown trip-
; count, to check the behavior of folding-the-tail (folding the remainder loop
; into the main loop using masking) together with interleaved-groups.
; When masked-interleave-group is disabled the interleave-groups will be
; invalidated during Legality check, so nothing to check here.
; When masked-interleave-group is enabled we check that there is no epilogue,
; and that the interleave-groups are vectorized using proper masking (with
; shuffling of the mask feeding the wide masked load/store).
; The mask itself is an And of two masks: one that masks away the remainder
; iterations, and one that masks away the 'else' of the 'if' statement.
;
; void masked_strided2_unknown_tc(const unsigned char* restrict p,
;                     unsigned char* restrict q,
;                     unsigned char guard,
;                     int n) {
; for(ix=0; ix < n; ++ix) {
;     if (ix > guard) {
;         char left = p[2*ix];
;         char right = p[2*ix + 1];
;         char max = max(left, right);
;         q[2*ix] = max;
;         q[2*ix+1] = 0 - max;
;     }
; }
;}

; ENABLED_MASKED_STRIDED-LABEL: @masked_strided2_unknown_tc(
; ENABLED_MASKED_STRIDED:       vector.body:
; ENABLED_MASKED_STRIDED-NEXT:    [[INDEX:%.*]] = phi i32 
; ENABLED_MASKED_STRIDED-NEXT:    [[VEC_IND:%.*]] = phi <8 x i32> [ <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP0:%.*]] = icmp sgt <8 x i32> [[VEC_IND]], {{.*}} 
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP1:%.*]] = shl nuw nsw i32 [[INDEX]], 1
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP2:%.*]] = getelementptr inbounds i8, i8* [[P:%.*]], i32 [[TMP1]]
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP3:%.*]] = icmp ule <8 x i32> [[VEC_IND]], {{.*}} 
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP4:%.*]] = and <8 x i1> [[TMP0]], [[TMP3]]
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP5:%.*]] = bitcast i8* [[TMP2]] to <16 x i8>*
; ENABLED_MASKED_STRIDED-NEXT:    [[INTERLEAVED_MASK:%.*]] = shufflevector <8 x i1> [[TMP4]], <8 x i1> undef, <16 x i32> <i32 0, i32 0, i32 1, i32 1, i32 2, i32 2, i32 3, i32 3, i32 4, i32 4, i32 5, i32 5, i32 6, i32 6, i32 7, i32 7>
; ENABLED_MASKED_STRIDED-NEXT:    [[WIDE_MASKED_VEC:%.*]] = call <16 x i8> @llvm.masked.load.v16i8.p0v16i8(<16 x i8>* [[TMP5]], i32 1, <16 x i1> [[INTERLEAVED_MASK]], <16 x i8> undef)
; ENABLED_MASKED_STRIDED-NEXT:    [[STRIDED_VEC:%.*]] = shufflevector <16 x i8> [[WIDE_MASKED_VEC]], <16 x i8> undef, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
; ENABLED_MASKED_STRIDED-NEXT:    [[STRIDED_VEC3:%.*]] = shufflevector <16 x i8> [[WIDE_MASKED_VEC]], <16 x i8> undef, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP6:%.*]] = or i32 [[TMP1]], 1
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP7:%.*]] = icmp slt <8 x i8> [[STRIDED_VEC]], [[STRIDED_VEC3]]
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP8:%.*]] = select <8 x i1> [[TMP7]], <8 x i8> [[STRIDED_VEC3]], <8 x i8> [[STRIDED_VEC]]
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP9:%.*]] = sub <8 x i8> zeroinitializer, [[TMP8]]
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP10:%.*]] = getelementptr inbounds i8, i8* [[Q:%.*]], i32 -1
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP11:%.*]] = getelementptr inbounds i8, i8* [[TMP10]], i32 [[TMP6]]
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP12:%.*]] = bitcast i8* [[TMP11]] to <16 x i8>*
; ENABLED_MASKED_STRIDED-NEXT:    [[INTERLEAVED_VEC:%.*]] = shufflevector <8 x i8> [[TMP8]], <8 x i8> [[TMP9]], <16 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11, i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
; ENABLED_MASKED_STRIDED-NEXT:    call void @llvm.masked.store.v16i8.p0v16i8(<16 x i8> [[INTERLEAVED_VEC]], <16 x i8>* [[TMP12]], i32 1, <16 x i1> [[INTERLEAVED_MASK]])
; ENABLED_MASKED_STRIDED-NEXT:    {{.*}} = add i32 [[INDEX]], 8
; ENABLED_MASKED_STRIDED-NEXT:    {{.*}} = add <8 x i32> {{.*}}, <i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8>
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP13:%.*]] = icmp eq i32 {{.*}}, {{.*}} 
; ENABLED_MASKED_STRIDED-NEXT:    br i1 [[TMP13]], 
; ENABLED_MASKED_STRIDED:       for.end:
; ENABLED_MASKED_STRIDED-NEXT:    ret void

define dso_local void @masked_strided2_unknown_tc(i8* noalias nocapture readonly %p, i8* noalias nocapture %q, i32 %guard, i32 %n) local_unnamed_addr optsize {
entry:
  %cmp22 = icmp sgt i32 %n, 0
  br i1 %cmp22, label %for.body.preheader, label %for.end

for.body.preheader:
  br label %for.body

for.body:
  %ix.023 = phi i32 [ %inc, %for.inc ], [ 0, %for.body.preheader ]
  %cmp1 = icmp sgt i32 %ix.023, %guard
  br i1 %cmp1, label %if.then, label %for.inc

if.then:
  %mul = shl nuw nsw i32 %ix.023, 1
  %arrayidx = getelementptr inbounds i8, i8* %p, i32 %mul
  %0 = load i8, i8* %arrayidx, align 1
  %add = or i32 %mul, 1
  %arrayidx3 = getelementptr inbounds i8, i8* %p, i32 %add
  %1 = load i8, i8* %arrayidx3, align 1
  %cmp.i = icmp slt i8 %0, %1
  %spec.select.i = select i1 %cmp.i, i8 %1, i8 %0
  %arrayidx5 = getelementptr inbounds i8, i8* %q, i32 %mul
  store i8 %spec.select.i, i8* %arrayidx5, align 1
  %sub = sub i8 0, %spec.select.i
  %arrayidx9 = getelementptr inbounds i8, i8* %q, i32 %add
  store i8 %sub, i8* %arrayidx9, align 1
  br label %for.inc

for.inc:
  %inc = add nuw nsw i32 %ix.023, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}

; Full groups under Optsize scenario again, with unknown trip-count, again in
; order to check the behavior of folding-the-tail (folding the remainder loop
; into the main loop using masking) together with interleaved-groups.
; This time the accesses are not conditional, they become conditional only
; due to tail folding.
; When masked-interleave-group is disabled the interleave-groups will be
; invalidated during cost-model checks, so we check for no epilogue and
; scalarized conditional accesses.
; When masked-interleave-group is enabled we check for no epilogue,
; and interleave-groups vectorized using proper masking (with
; shuffling of the mask feeding the wide masked load/store).
; (Same vectorization scheme as for the previous loop with conditional accesses
; except here the mask only masks away the remainder iterations.)
;
; void unconditional_masked_strided2_unknown_tc(const unsigned char* restrict p,
;                     unsigned char* restrict q,
;                     int n) {
; for(ix=0; ix < n; ++ix) {
;         char left = p[2*ix];
;         char right = p[2*ix + 1];
;         char max = max(left, right);
;         q[2*ix] = max;
;         q[2*ix+1] = 0 - max;
; }
;}

; DISABLED_MASKED_STRIDED-LABEL: @unconditional_masked_strided2_unknown_tc(
; DISABLED_MASKED_STRIDED:       vector.body:
; DISABLED_MASKED_STRIDED-NEXT:    [[INDEX:%.*]] = phi i32 
; DISABLED_MASKED_STRIDED-NEXT:    [[VEC_IND:%.*]] = phi <8 x i32> [ <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; DISABLED_MASKED_STRIDED-NEXT:    [[TMP0:%.*]] = shl nuw nsw <8 x i32> [[VEC_IND]], <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
; DISABLED_MASKED_STRIDED-NEXT:    [[TMP1:%.*]] = icmp ule <8 x i32> [[VEC_IND]], {{.*}}
; DISABLED_MASKED_STRIDED-NEXT:    [[TMP2:%.*]] = extractelement <8 x i1> [[TMP1]], i32 0
; DISABLED_MASKED_STRIDED-NEXT:    br i1 [[TMP2]], label [[PRED_LOAD_IF:%.*]], label [[PRED_LOAD_CONTINUE:%.*]]
; DISABLED_MASKED_STRIDED:       pred.load.if:
; DISABLED_MASKED_STRIDED-NEXT:    [[TMP3:%.*]] = extractelement <8 x i32> [[TMP0]], i32 0
; DISABLED_MASKED_STRIDED-NEXT:    [[TMP4:%.*]] = getelementptr inbounds i8, i8* [[P:%.*]], i32 [[TMP3]]
; DISABLED_MASKED_STRIDED-NEXT:    [[TMP5:%.*]] = load i8, i8* [[TMP4]], align 1
; DISABLED_MASKED_STRIDED-NEXT:    [[TMP6:%.*]] = insertelement <8 x i8> undef, i8 [[TMP5]], i32 0
; DISABLED_MASKED_STRIDED-NEXT:    br label [[PRED_LOAD_CONTINUE]]
; DISABLED_MASKED_STRIDED-NOT:   for.body:
; DISABLED_MASKED_STRIDED:       for.end:
; DISABLED_MASKED_STRIDED-NEXT:    ret void



; ENABLED_MASKED_STRIDED-LABEL: @unconditional_masked_strided2_unknown_tc(
; ENABLED_MASKED_STRIDED:       vector.body:
; ENABLED_MASKED_STRIDED-NEXT:    [[INDEX:%.*]] = phi i32 
; ENABLED_MASKED_STRIDED-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <8 x i32> undef, i32 [[INDEX]], i32 0
; ENABLED_MASKED_STRIDED-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <8 x i32> [[BROADCAST_SPLATINSERT]], <8 x i32> undef, <8 x i32> zeroinitializer
; ENABLED_MASKED_STRIDED-NEXT:    [[INDUCTION:%.*]] = add <8 x i32> [[BROADCAST_SPLAT]], <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP0:%.*]] = shl nuw nsw i32 [[INDEX]], 1
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP1:%.*]] = getelementptr inbounds i8, i8* [[P:%.*]], i32 [[TMP0]]
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP2:%.*]] = icmp ule <8 x i32> {{.*}}, {{.*}}
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP3:%.*]] = bitcast i8* [[TMP1]] to <16 x i8>*
; ENABLED_MASKED_STRIDED-NEXT:    [[INTERLEAVED_MASK:%.*]] = shufflevector <8 x i1> [[TMP2]], <8 x i1> undef, <16 x i32> <i32 0, i32 0, i32 1, i32 1, i32 2, i32 2, i32 3, i32 3, i32 4, i32 4, i32 5, i32 5, i32 6, i32 6, i32 7, i32 7>
; ENABLED_MASKED_STRIDED-NEXT:    [[WIDE_MASKED_VEC:%.*]] = call <16 x i8> @llvm.masked.load.v16i8.p0v16i8(<16 x i8>* [[TMP3]], i32 1, <16 x i1> [[INTERLEAVED_MASK]], <16 x i8> undef)
; ENABLED_MASKED_STRIDED-NEXT:    [[STRIDED_VEC:%.*]] = shufflevector <16 x i8> [[WIDE_MASKED_VEC]], <16 x i8> undef, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
; ENABLED_MASKED_STRIDED-NEXT:    [[STRIDED_VEC3:%.*]] = shufflevector <16 x i8> [[WIDE_MASKED_VEC]], <16 x i8> undef, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP4:%.*]] = or i32 [[TMP0]], 1
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP5:%.*]] = icmp slt <8 x i8> [[STRIDED_VEC]], [[STRIDED_VEC3]]
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP6:%.*]] = select <8 x i1> [[TMP5]], <8 x i8> [[STRIDED_VEC3]], <8 x i8> [[STRIDED_VEC]]
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP7:%.*]] = sub <8 x i8> zeroinitializer, [[TMP6]]
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP8:%.*]] = getelementptr inbounds i8, i8* [[Q:%.*]], i32 -1
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP9:%.*]] = getelementptr inbounds i8, i8* [[TMP8]], i32 [[TMP4]]
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP10:%.*]] = bitcast i8* [[TMP9]] to <16 x i8>*
; ENABLED_MASKED_STRIDED-NEXT:    [[INTERLEAVED_VEC:%.*]] = shufflevector <8 x i8> [[TMP6]], <8 x i8> [[TMP7]], <16 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11, i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
; ENABLED_MASKED_STRIDED-NEXT:    call void @llvm.masked.store.v16i8.p0v16i8(<16 x i8> [[INTERLEAVED_VEC]], <16 x i8>* [[TMP10]], i32 1, <16 x i1> [[INTERLEAVED_MASK]])
; ENABLED_MASKED_STRIDED-NEXT:    {{.*}} = add i32 [[INDEX]], 8
; ENABLED_MASKED_STRIDED-NEXT:    [[TMP11:%.*]] = icmp eq i32 {{.*}}, {{.*}}
; ENABLED_MASKED_STRIDED-NEXT:    br i1 [[TMP11]]
; ENABLED_MASKED_STRIDED:       for.end:
; ENABLED_MASKED_STRIDED-NEXT:    ret void

define dso_local void @unconditional_masked_strided2_unknown_tc(i8* noalias nocapture readonly %p, i8* noalias nocapture %q, i32 %n) local_unnamed_addr optsize {
entry:
  %cmp20 = icmp sgt i32 %n, 0
  br i1 %cmp20, label %for.body.preheader, label %for.end

for.body.preheader:
  br label %for.body

for.body:
  %ix.021 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %mul = shl nuw nsw i32 %ix.021, 1
  %arrayidx = getelementptr inbounds i8, i8* %p, i32 %mul
  %0 = load i8, i8* %arrayidx, align 1
  %add = or i32 %mul, 1
  %arrayidx2 = getelementptr inbounds i8, i8* %p, i32 %add
  %1 = load i8, i8* %arrayidx2, align 1
  %cmp.i = icmp slt i8 %0, %1
  %spec.select.i = select i1 %cmp.i, i8 %1, i8 %0
  %arrayidx4 = getelementptr inbounds i8, i8* %q, i32 %mul
  store i8 %spec.select.i, i8* %arrayidx4, align 1
  %sub = sub i8 0, %spec.select.i
  %arrayidx8 = getelementptr inbounds i8, i8* %q, i32 %add
  store i8 %sub, i8* %arrayidx8, align 1
  %inc = add nuw nsw i32 %ix.021, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}

