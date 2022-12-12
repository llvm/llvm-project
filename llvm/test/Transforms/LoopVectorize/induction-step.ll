; RUN: opt < %s -passes=loop-vectorize -force-vector-interleave=1 -force-vector-width=8 -S | FileCheck %s

; int int_inc;
;
;int induction_with_global(int init, int *restrict A, int N) {
;  int x = init;
;  for (int i=0;i<N;i++){
;    A[i] = x;
;    x += int_inc;
;  }
;  return x;
;}

; CHECK-LABEL: @induction_with_global(
; CHECK:       for.body.lr.ph:
; CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr @int_inc, align 4
; CHECK:       vector.ph:
; CHECK:         [[DOTSPLATINSERT:%.*]] = insertelement <8 x i32> poison, i32 %init, i32 0
; CHECK-NEXT:    [[DOTSPLAT:%.*]] = shufflevector <8 x i32> [[DOTSPLATINSERT]], <8 x i32> poison, <8 x i32> zeroinitializer
; CHECK-NEXT:    [[DOTSPLATINSERT2:%.*]] = insertelement <8 x i32> poison, i32 [[TMP0]], i32 0
; CHECK-NEXT:    [[DOTSPLAT3:%.*]] = shufflevector <8 x i32> [[DOTSPLATINSERT2]], <8 x i32> poison, <8 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP6:%.*]] = mul <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, [[DOTSPLAT3]]
; CHECK-NEXT:    [[INDUCTION4:%.*]] = add <8 x i32> [[DOTSPLAT]], [[TMP6]]
; CHECK-NEXT:    [[TMP7:%.*]] = mul i32 [[TMP0]], 8
; CHECK-NEXT:    [[DOTSPLATINSERT5:%.*]] = insertelement <8 x i32> poison, i32 [[TMP7]], i32 0
; CHECK-NEXT:    [[DOTSPLAT6:%.*]] = shufflevector <8 x i32> [[DOTSPLATINSERT5]], <8 x i32> poison, <8 x i32> zeroinitializer
; CHECK-NEXT:    br label %vector.body
; CHECK:       vector.body:
; CHECK-NEXT:    %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; CHECK-NEXT:    %vec.ind = phi <8 x i32> [ [[INDUCTION4]], %vector.ph ], [ %vec.ind.next, %vector.body ]
; CHECK:         [[TMP8:%.*]] = add i64 %index, 0
; CHECK-NEXT:    [[TMP9:%.*]] = getelementptr inbounds i32, ptr [[A:%.*]], i64 [[TMP8]]
; CHECK-NEXT:    [[TMP10:%.*]] = getelementptr inbounds i32, ptr [[TMP9]], i32 0
; CHECK-NEXT:    store <8 x i32> %vec.ind, ptr [[TMP10]], align 4
; CHECK:         %index.next = add nuw i64 %index, 8
; CHECK-NEXT:    %vec.ind.next = add <8 x i32> %vec.ind, [[DOTSPLAT6]]
; CHECK:         br i1 {{.*}}, label %middle.block, label %vector.body

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"


@int_inc = common global i32 0, align 4

define i32 @induction_with_global(i32 %init, ptr noalias nocapture %A, i32 %N) {
entry:
  %cmp4 = icmp sgt i32 %N, 0
  br i1 %cmp4, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  %0 = load i32, ptr @int_inc, align 4
  %1 = mul i32 %0, %N
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  %indvars.iv = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next, %for.body ]
  %x.05 = phi i32 [ %init, %for.body.lr.ph ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  store i32 %x.05, ptr %arrayidx, align 4
  %add = add nsw i32 %0, %x.05
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %N
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:                                 ; preds = %for.body
  %2 = add i32 %1, %init
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  %x.0.lcssa = phi i32 [ %init, %entry ], [ %2, %for.end.loopexit ]
  ret i32 %x.0.lcssa
}


;int induction_with_loop_inv(int init, int *restrict A, int N, int M) {
;  int x = init;
;  for (int j = 0; j < M; j++) {
;    for (int i=0; i<N; i++){
;      A[i] = x;
;      x += j; // induction step is a loop invariant variable
;    }
;  }
;  return x;
;}

; CHECK-LABEL: @induction_with_loop_inv(
; CHECK:       vector.ph:
; CHECK:         [[DOTSPLATINSERT:%.*]] = insertelement <8 x i32> poison, i32 %x.011, i32 0
; CHECK-NEXT:    [[DOTSPLAT:%.*]] = shufflevector <8 x i32> [[DOTSPLATINSERT]], <8 x i32> poison, <8 x i32> zeroinitializer
; CHECK-NEXT:    [[DOTSPLATINSERT2:%.*]] = insertelement <8 x i32> poison, i32 %j.012, i32 0
; CHECK-NEXT:    [[DOTSPLAT3:%.*]] = shufflevector <8 x i32> [[DOTSPLATINSERT2]], <8 x i32> poison, <8 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP4:%.*]] = mul <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, [[DOTSPLAT3]]
; CHECK-NEXT:    [[INDUCTION4:%.*]] = add <8 x i32> [[DOTSPLAT]], [[TMP4]]
; CHECK-NEXT:    [[TMP5:%.*]] = mul i32 %j.012, 8
; CHECK-NEXT:    [[DOTSPLATINSERT5:%.*]] = insertelement <8 x i32> poison, i32 [[TMP5]], i32 0
; CHECK-NEXT:    [[DOTSPLAT6:%.*]] = shufflevector <8 x i32> [[DOTSPLATINSERT5]], <8 x i32> poison, <8 x i32> zeroinitializer
; CHECK-NEXT:    br label %vector.body
; CHECK:       vector.body:
; CHECK-NEXT:    %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; CHECK-NEXT:    %vec.ind = phi <8 x i32> [ [[INDUCTION4]], %vector.ph ], [ %vec.ind.next, %vector.body ]
; CHECK:         [[TMP6:%.*]] = add i64 %index, 0
; CHECK-NEXT:    [[TMP7:%.*]] = getelementptr inbounds i32, ptr [[A:%.*]], i64 [[TMP6]]
; CHECK-NEXT:    [[TMP8:%.*]] = getelementptr inbounds i32, ptr [[TMP7]], i32 0
; CHECK-NEXT:    store <8 x i32> %vec.ind, ptr [[TMP8]], align 4
; CHECK:         %index.next = add nuw i64 %index, 8
; CHECK-NEXT:    %vec.ind.next = add <8 x i32> %vec.ind, [[DOTSPLAT6]]
; CHECK:         br i1 {{.*}}, label %middle.block, label %vector.body

define i32 @induction_with_loop_inv(i32 %init, ptr noalias nocapture %A, i32 %N, i32 %M) {
entry:
  %cmp10 = icmp sgt i32 %M, 0
  br i1 %cmp10, label %for.cond1.preheader.lr.ph, label %for.end6

for.cond1.preheader.lr.ph:                        ; preds = %entry
  %cmp27 = icmp sgt i32 %N, 0
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.inc4, %for.cond1.preheader.lr.ph
  %indvars.iv15 = phi i32 [ 0, %for.cond1.preheader.lr.ph ], [ %indvars.iv.next16, %for.inc4 ]
  %j.012 = phi i32 [ 0, %for.cond1.preheader.lr.ph ], [ %inc5, %for.inc4 ]
  %x.011 = phi i32 [ %init, %for.cond1.preheader.lr.ph ], [ %x.1.lcssa, %for.inc4 ]
  br i1 %cmp27, label %for.body3.preheader, label %for.inc4

for.body3.preheader:                              ; preds = %for.cond1.preheader
  br label %for.body3

for.body3:                                        ; preds = %for.body3.preheader, %for.body3
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body3 ], [ 0, %for.body3.preheader ]
  %x.18 = phi i32 [ %add, %for.body3 ], [ %x.011, %for.body3.preheader ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  store i32 %x.18, ptr %arrayidx, align 4
  %add = add nsw i32 %x.18, %j.012
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %N
  br i1 %exitcond, label %for.inc4.loopexit, label %for.body3

for.inc4.loopexit:                                ; preds = %for.body3
  %0 = add i32 %x.011, %indvars.iv15
  br label %for.inc4

for.inc4:                                         ; preds = %for.inc4.loopexit, %for.cond1.preheader
  %x.1.lcssa = phi i32 [ %x.011, %for.cond1.preheader ], [ %0, %for.inc4.loopexit ]
  %inc5 = add nuw nsw i32 %j.012, 1
  %indvars.iv.next16 = add i32 %indvars.iv15, %N
  %exitcond17 = icmp eq i32 %inc5, %M
  br i1 %exitcond17, label %for.end6.loopexit, label %for.cond1.preheader

for.end6.loopexit:                                ; preds = %for.inc4
  %x.1.lcssa.lcssa = phi i32 [ %x.1.lcssa, %for.inc4 ]
  br label %for.end6

for.end6:                                         ; preds = %for.end6.loopexit, %entry
  %x.0.lcssa = phi i32 [ %init, %entry ], [ %x.1.lcssa.lcssa, %for.end6.loopexit ]
  ret i32 %x.0.lcssa
}


; CHECK-LABEL: @non_primary_iv_loop_inv_trunc(
; CHECK:       vector.ph:
; CHECK:         [[TMP3:%.*]] = trunc i64 %step to i32
; CHECK-NEXT:    [[DOTSPLATINSERT5:%.*]] = insertelement <8 x i32> poison, i32 [[TMP3]], i32 0
; CHECK-NEXT:    [[DOTSPLAT6:%.*]] = shufflevector <8 x i32> [[DOTSPLATINSERT5]], <8 x i32> poison, <8 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP4:%.*]] = mul <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, [[DOTSPLAT6]]
; CHECK-NEXT:    [[INDUCTION7:%.*]] = add <8 x i32> zeroinitializer, [[TMP4]]
; CHECK-NEXT:    [[TMP5:%.*]] = mul i32 [[TMP3]], 8
; CHECK-NEXT:    [[DOTSPLATINSERT8:%.*]] = insertelement <8 x i32> poison, i32 [[TMP5]], i32 0
; CHECK-NEXT:    [[DOTSPLAT9:%.*]] = shufflevector <8 x i32> [[DOTSPLATINSERT8]], <8 x i32> poison, <8 x i32> zeroinitializer
; CHECK-NEXT:    br label %vector.body
; CHECK:       vector.body:
; CHECK-NEXT:    %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; CHECK:         [[VEC_IND10:%.*]] = phi <8 x i32> [ [[INDUCTION7]], %vector.ph ], [ [[VEC_IND_NEXT11:%.*]], %vector.body ]
; CHECK:         [[TMP6:%.*]] = add i64 %index, 0
; CHECK-NEXT:    [[TMP7:%.*]] = getelementptr inbounds i32, ptr [[A:%.*]], i64 [[TMP6]]
; CHECK-NEXT:    [[TMP8:%.*]] = getelementptr inbounds i32, ptr [[TMP7]], i32 0
; CHECK-NEXT:    store <8 x i32> [[VEC_IND10]], ptr [[TMP8]], align 4
; CHECK-NEXT:    %index.next = add nuw i64 %index, 8
; CHECK:         [[VEC_IND_NEXT11]] = add <8 x i32> [[VEC_IND10]], [[DOTSPLAT9]]
; CHECK:         br i1 {{.*}}, label %middle.block, label %vector.body

define void @non_primary_iv_loop_inv_trunc(ptr %a, i64 %n, i64 %step) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ %i.next, %for.body ], [ 0, %entry ]
  %j = phi i64 [ %j.next, %for.body ], [ 0, %entry ]
  %tmp0 = getelementptr inbounds i32, ptr %a, i64 %i
  %tmp1 = trunc i64 %j to i32
  store i32 %tmp1, ptr %tmp0, align 4
  %i.next = add nuw nsw i64 %i, 1
  %j.next = add nuw nsw i64 %j, %step
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

; CHECK-LABEL: @iv_no_binary_op_in_descriptor(
; CHECK:       vector.ph:
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_IND:%.*]] = phi <8 x i64> [ <i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7>, %vector.ph ], [ [[VEC_IND_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds i64, ptr [[DST:%.*]], i64 [[TMP0]]
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr inbounds i64, ptr [[TMP1]], i32 0
; CHECK-NEXT:    store <8 x i64> [[VEC_IND]], ptr [[TMP2]], align 8
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 8
; CHECK-NEXT:    [[VEC_IND_NEXT]] = add <8 x i64> [[VEC_IND]], <i64 8, i64 8, i64 8, i64 8, i64 8, i64 8, i64 8, i64 8>
; CHECK-NEXT:    [[TMP3:%.*]] = icmp eq i64 [[INDEX_NEXT]], 1000
; CHECK-NEXT:    br i1 [[TMP3]], label %middle.block, label [[VECTOR_BODY]]

define void @iv_no_binary_op_in_descriptor(i1 %c, ptr %dst) {
entry:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.next.p, %loop.latch ]
  %gep = getelementptr inbounds i64, ptr %dst, i64 %iv
  store i64 %iv, ptr %gep, align 8
  %iv.next = add i64 %iv, 1
  br label %loop.latch

loop.latch:
  %iv.next.p = phi i64 [ %iv.next, %loop.header ]
  %exitcond.not = icmp eq i64 %iv.next.p, 1000
  br i1 %exitcond.not, label %exit, label %loop.header

exit:
  ret void
}
