; RUN: opt -loop-vectorize -scalable-vectorization=off -force-vector-width=4 -prefer-predicate-over-epilogue=predicate-dont-vectorize -S < %s | FileCheck %s

; NOTE: These tests aren't really target-specific, but it's convenient to target AArch64
; so that TTI.isLegalMaskedLoad can return true.

target triple = "aarch64-linux-gnu"

; The original loop had an unconditional uniform load. Let's make sure
; we don't artificially create new predicated blocks for the load.
define void @uniform_load(i32* noalias %dst, i32* noalias readonly %src, i64 %n) #0 {
; CHECK-LABEL: @uniform_load(
; CHECK:       vector.body:
; CHECK-NEXT:    [[IDX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[IDX_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[TMP3:%.*]] = add i64 [[IDX]], 0
; CHECK-NEXT:    [[LOOP_PRED:%.*]] = call <4 x i1> @llvm.get.active.lane.mask.v4i1.i64(i64 [[TMP3]], i64 %n)
; CHECK-NEXT:    [[LOAD_VAL:%.*]] = load i32, i32* %src, align 4
; CHECK-NOT:     load i32, i32* %src, align 4
; CHECK-NEXT:    [[TMP4:%.*]] = insertelement <4 x i32> poison, i32 [[LOAD_VAL]], i32 0
; CHECK-NEXT:    [[TMP5:%.*]] = shufflevector <4 x i32> [[TMP4]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP6:%.*]] = getelementptr inbounds i32, i32* %dst, i64 [[TMP3]]
; CHECK-NEXT:    [[TMP7:%.*]] = getelementptr inbounds i32, i32* [[TMP6]], i32 0
; CHECK-NEXT:    [[STORE_PTR:%.*]] = bitcast i32* [[TMP7]] to <4 x i32>*
; CHECK-NEXT:    call void @llvm.masked.store.v4i32.p0v4i32(<4 x i32> [[TMP5]], <4 x i32>* [[STORE_PTR]], i32 4, <4 x i1> [[LOOP_PRED]])
; CHECK-NEXT:    [[IDX_NEXT]] = add i64 [[IDX]], 4
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i64 [[IDX_NEXT]], %n.vec
; CHECK-NEXT:    br i1 [[CMP]], label %middle.block, label %vector.body

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
; CHECK-NEXT:    [[TMP3:%.*]] = add i64 [[IDX]], 0
; CHECK-NEXT:    [[LOOP_PRED:%.*]] = call <4 x i1> @llvm.get.active.lane.mask.v4i1.i64(i64 [[TMP3]], i64 %n)
; CHECK:         [[COND_LOAD:%.*]] = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* {{%.*}}, i32 4, <4 x i1> [[LOOP_PRED]], <4 x i32> poison)
; CHECK-NEXT:    [[TMP4:%.*]] = icmp eq <4 x i32> [[COND_LOAD]], zeroinitializer
; CHECK-NEXT:    [[TMP5:%.*]] = xor <4 x i1> [[TMP4]], <i1 true, i1 true, i1 true, i1 true>
; CHECK-NEXT:    [[MASK:%.*]] = select <4 x i1> [[LOOP_PRED]], <4 x i1> [[TMP5]], <4 x i1> zeroinitializer
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

attributes #0 = { "target-features"="+neon,+sve,+v8.1a" vscale_range(2, 0) }
