; RUN: opt -loop-vectorize -scalable-vectorization=off -force-vector-width=4 -prefer-predicate-over-epilogue=predicate-dont-vectorize -S < %s | FileCheck %s

; NOTE: These tests aren't really target-specific, but it's convenient to target AArch64
; so that TTI.isLegalMaskedLoad can return true.

target triple = "aarch64-linux-gnu"

define void @uniform_load(i32* noalias %dst, i32* noalias readonly %src, i64 %n) #0 {
; CHECK-LABEL: @uniform_load(
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %pred.load.continue8 ]
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT1:%.*]] = insertelement <4 x i64> poison, i64 [[INDEX]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT2:%.*]] = shufflevector <4 x i64> [[BROADCAST_SPLATINSERT1]], <4 x i64> poison, <4 x i32> zeroinitializer
; CHECK-NEXT:    [[INDUCTION:%.*]] = add <4 x i64> [[BROADCAST_SPLAT2]], <i64 0, i64 1, i64 2, i64 3>
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP1:%.*]] = icmp ule <4 x i64> [[INDUCTION]]
; CHECK-NEXT:    [[TMP2:%.*]] = extractelement <4 x i1> [[TMP1]], i32 0
; CHECK-NEXT:    br i1 [[TMP2]], label [[PRED_LOAD_IF:%.*]], label %pred.load.continue
; CHECK:       pred.load.if:
; CHECK-NEXT:    [[TMP3:%.*]] = load i32, i32* [[SRC:%.*]], align 4
; CHECK-NEXT:    [[TMP4:%.*]] = insertelement <4 x i32> poison, i32 [[TMP3]], i32 0
; CHECK-NEXT:    br label %pred.load.continue
; CHECK:       pred.load.continue:
; CHECK-NEXT:    [[TMP5:%.*]] = phi <4 x i32> [ poison, %vector.body ], [ [[TMP4]], [[PRED_LOAD_IF]] ]
; CHECK-NEXT:    [[TMP6:%.*]] = extractelement <4 x i1> [[TMP1]], i32 1
; CHECK-NEXT:    br i1 [[TMP6]], label [[PRED_LOAD_IF3:%.*]], label %pred.load.continue4
; CHECK:       pred.load.if3:
; CHECK-NEXT:    [[TMP7:%.*]] = load i32, i32* [[SRC]], align 4
; CHECK-NEXT:    [[TMP8:%.*]] = insertelement <4 x i32> [[TMP5]], i32 [[TMP7]], i32 1
; CHECK-NEXT:    br label %pred.load.continue4
; CHECK:       pred.load.continue4:
; CHECK-NEXT:    [[TMP9:%.*]] = phi <4 x i32> [ [[TMP5]], %pred.load.continue ], [ [[TMP8]], %pred.load.if3 ]
; CHECK-NEXT:    [[TMP10:%.*]] = extractelement <4 x i1> [[TMP1]], i32 2
; CHECK-NEXT:    br i1 [[TMP10]], label [[PRED_LOAD_IF5:%.*]], label %pred.load.continue6
; CHECK:       pred.load.if5:
; CHECK-NEXT:    [[TMP11:%.*]] = load i32, i32* [[SRC]], align 4
; CHECK-NEXT:    [[TMP12:%.*]] = insertelement <4 x i32> [[TMP9]], i32 [[TMP11]], i32 2
; CHECK-NEXT:    br label %pred.load.continue6
; CHECK:       pred.load.continue6:
; CHECK-NEXT:    [[TMP13:%.*]] = phi <4 x i32> [ [[TMP9]], %pred.load.continue4 ], [ [[TMP12]], %pred.load.if5 ]
; CHECK-NEXT:    [[TMP14:%.*]] = extractelement <4 x i1> [[TMP1]], i32 3
; CHECK-NEXT:    br i1 [[TMP14]], label [[PRED_LOAD_IF7:%.*]], label %pred.load.continue8
; CHECK:       pred.load.if7:
; CHECK-NEXT:    [[TMP15:%.*]] = load i32, i32* [[SRC]], align 4
; CHECK-NEXT:    [[TMP16:%.*]] = insertelement <4 x i32> [[TMP13]], i32 [[TMP15]], i32 3
; CHECK-NEXT:    br label %pred.load.continue8
; CHECK:       pred.load.continue8:
; CHECK-NEXT:    [[TMP17:%.*]] = phi <4 x i32> [ [[TMP13]], %pred.load.continue6 ], [ [[TMP16]], [[PRED_LOAD_IF7]] ]
; CHECK-NEXT:    [[TMP18:%.*]] = getelementptr inbounds i32, i32* [[DST:%.*]], i64 [[TMP0]]
; CHECK-NEXT:    [[TMP19:%.*]] = getelementptr inbounds i32, i32* [[TMP18]], i32 0
; CHECK-NEXT:    [[TMP20:%.*]] = bitcast i32* [[TMP19]] to <4 x i32>*
; CHECK-NEXT:    call void @llvm.masked.store.v4i32.p0v4i32(<4 x i32> [[TMP17]], <4 x i32>* [[TMP20]], i32 4, <4 x i1> [[TMP1]])
; CHECK-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], 4
; CHECK-NEXT:    [[TMP21:%.*]] = icmp eq i64 [[INDEX_NEXT]], %n.vec
; CHECK-NEXT:    br i1 [[TMP21]], label [[MIDDLE_BLOCK:%.*]], label %vector.body

entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %val = load i32, i32* %src, align 4
  %arrayidx = getelementptr inbounds i32, i32* %dst, i64 %indvars.iv
  store i32 %val, i32* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; The original loop had a conditional uniform load. In this case we actually
; do need to perform conditional loads and so we end up using a gather instead.
; However, we at least ensure the mask is the overlap of the loop predicate
; and the original condition.
define void @cond_uniform_load(i32* nocapture %dst, i32* nocapture readonly %src, i32* nocapture readonly %cond, i64 %n) #0 {
; CHECK-LABEL: @cond_uniform_load(
; CHECK:       vector.ph:
; CHECK:         [[TMP1:%.*]] = insertelement <4 x i32*> poison, i32* %src, i32 0
; CHECK-NEXT:    [[SRC_SPLAT:%.*]] = shufflevector <4 x i32*> [[TMP1]], <4 x i32*> poison, <4 x i32> zeroinitializer
; CHECK:       vector.body:
; CHECK-NEXT:    [[IDX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[IDX_NEXT:%.*]], %vector.body ]
; CHECK:         [[TMP1:%.*]] = insertelement <4 x i64> poison, i64 [[IDX]], i32 0
; CHECK-NEXT:    [[TMP2:%.*]] = shufflevector <4 x i64> [[TMP1]], <4 x i64> poison, <4 x i32> zeroinitializer
; CHECK-NEXT:    [[INDUCTION:%.*]] = add <4 x i64> [[TMP2]], <i64 0, i64 1, i64 2, i64 3>
; CHECK:         [[LOOP_PRED:%.*]] = icmp ule <4 x i64> [[INDUCTION]]
; CHECK:         [[COND_LOAD:%.*]] = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* {{%.*}}, i32 4, <4 x i1> [[LOOP_PRED]], <4 x i32> poison)
; CHECK-NEXT:    [[TMP3:%.*]] = icmp eq <4 x i32> [[COND_LOAD]], zeroinitializer
; CHECK-NEXT:    [[TMP4:%.*]] = xor <4 x i1> [[TMP3]], <i1 true, i1 true, i1 true, i1 true>
; CHECK-NEXT:    [[MASK:%.*]] = select <4 x i1> [[LOOP_PRED]], <4 x i1> [[TMP4]], <4 x i1> zeroinitializer
; CHECK-NEXT:    call <4 x i32> @llvm.masked.gather.v4i32.v4p0i32(<4 x i32*> [[SRC_SPLAT]], i32 4, <4 x i1> [[MASK]], <4 x i32> undef)
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %if.end
  %index = phi i64 [ %index.next, %if.end ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %cond, i64 %index
  %0 = load i32, i32* %arrayidx, align 4
  %tobool.not = icmp eq i32 %0, 0
  br i1 %tobool.not, label %if.end, label %if.then

if.then:                                          ; preds = %for.body
  %1 = load i32, i32* %src, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body
  %val.0 = phi i32 [ %1, %if.then ], [ 0, %for.body ]
  %arrayidx1 = getelementptr inbounds i32, i32* %dst, i64 %index
  store i32 %val.0, i32* %arrayidx1, align 4
  %index.next = add nuw i64 %index, 1
  %exitcond.not = icmp eq i64 %index.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:                                          ; preds = %for.inc, %entry
  ret void
}

attributes #0 = { "target-features"="+neon,+sve,+v8.1a" }
