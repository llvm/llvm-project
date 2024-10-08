; RUN: opt < %s -passes="loop-vectorize" -force-vector-interleave=1 -force-vector-width=4 -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; This test checks that we can vectorize loop with reduction variable
; stored in an invariant address.
;
; int sum = 0;
; for(i=0..N) {
;   sum += src[i];
;   dst[42] = sum;
; }
; CHECK-LABEL: @reduc_store
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH:%.*]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY:%.*]] ]
; CHECK-NEXT:    [[VEC_PHI:%.*]] = phi <4 x i32> [ zeroinitializer, [[VECTOR_PH]] ], [ [[TMP4:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds i32, ptr [[SRC:%.*]], i64 [[TMP0]]
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr inbounds i32, ptr [[TMP1]], i32 0
; CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <4 x i32>, ptr [[TMP2]], align 4
; CHECK-NEXT:    [[TMP4]] = add <4 x i32> [[VEC_PHI]], [[WIDE_LOAD]]
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 4
; CHECK-NEXT:    [[TMP5:%.*]] = icmp eq i64 [[INDEX_NEXT]], 1000
; CHECK-NEXT:    br i1 [[TMP5]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP3:![0-9]+]]
; CHECK:       middle.block:
; CHECK-NEXT:    [[TMP6:%.*]] = call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> [[TMP4]])
; CHECK-NEXT:    store i32 [[TMP6]], ptr [[GEP_DST:%.*]], align 4
; CHECK-NEXT:    br i1 true, label [[EXIT:%.*]], label [[SCALAR_PH:%.*]]
define void @reduc_store(ptr %dst, ptr readonly %src) {
entry:
  %gep.dst = getelementptr inbounds i32, ptr %dst, i64 42
  store i32 0, ptr %gep.dst, align 4
  br label %for.body

for.body:
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %gep.src = getelementptr inbounds i32, ptr %src, i64 %iv
  %0 = load i32, ptr %gep.src, align 4
  %add = add nsw i32 %sum, %0
  store i32 %add, ptr %gep.dst, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %exit, label %for.body

exit:
  ret void
}

; Same as above but with floating point numbers instead.
;
; float sum = 0;
; for(i=0..N) {
;   sum += src[i];
;   dst[42] = sum;
; }
; CHECK-LABEL: @reduc_store_fadd_fast
; CHECK: vector.body:
; CHECK: phi <4 x float>
; CHECK: load <4 x float>
; CHECK: fadd fast <4 x float>
; CHECK-NOT: store float %{{[0-9]+}}, ptr %gep.dst
; CHECK: middle.block:
; CHECK-NEXT: [[TMP:%.*]] = call fast float @llvm.vector.reduce.fadd.v4f32
; CHECK-NEXT: store float %{{[0-9]+}}, ptr %gep.dst
define void @reduc_store_fadd_fast(ptr %dst, ptr readonly %src) {
entry:
  %gep.dst = getelementptr inbounds float, ptr %dst, i64 42
  store float 0.000000e+00, ptr %gep.dst, align 4
  br label %for.body

for.body:
  %sum = phi float [ 0.000000e+00, %entry ], [ %add, %for.body ]
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %gep.src = getelementptr inbounds float, ptr %src, i64 %iv
  %0 = load float, ptr %gep.src, align 4
  %add = fadd fast float %sum, %0
  store float %add, ptr %gep.dst, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %exit, label %for.body

exit:
  ret void
}

; Check that if we have a read from an invariant address, we do not vectorize.
;
; int sum = 0;
; for(i=0..N) {
;   sum += src[i];
;   dst.2[i] = dst[42];
;   dst[42] = sum;
; }
; CHECK-LABEL: @reduc_store_load
; CHECK-NOT: vector.body
define void @reduc_store_load(ptr %dst, ptr readonly %src, ptr noalias %dst.2) {
entry:
  %gep.dst = getelementptr inbounds i32, ptr %dst, i64 42
  store i32 0, ptr %gep.dst, align 4
  br label %for.body

for.body:
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %gep.src = getelementptr inbounds i32, ptr %src, i64 %iv
  %0 = load i32, ptr %gep.src, align 4
  %add = add nsw i32 %sum, %0
  %lv = load i32, ptr %gep.dst
  %gep.dst.2  = getelementptr inbounds i32, ptr %dst.2, i64 %iv
  store i32 %lv, ptr %gep.dst.2, align 4
  store i32 %add, ptr %gep.dst, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %exit, label %for.body

exit:
  ret void
}

; Check that if we have a read from an invariant address, we do not vectorize,
; even if we vectorize with runtime checks. The test below is a variant of
; @reduc_store_load with a non-constant dependence distance, resulting in
; vectorization with runtime checks.
;
; CHECK-LABEL: @reduc_store_load_with_non_constant_distance_dependence
; CHECK-NOT: vector.body:
define void @reduc_store_load_with_non_constant_distance_dependence(ptr %dst, ptr noalias %dst.2, i64 %off) {
entry:
  %gep.dst = getelementptr inbounds i32, ptr %dst, i64 42
  %dst.2.off = getelementptr inbounds i32, ptr %dst.2, i64 %off
  store i32 0, ptr %gep.dst, align 4
  br label %for.body

for.body:
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %gep.src = getelementptr inbounds i32, ptr %dst.2, i64 %iv
  %0 = load i32, ptr %gep.src, align 4
  %iv.off = mul i64 %iv, 2
  %add = add nsw i32 %sum, %0
  %lv = load i32, ptr %gep.dst
  store i32 %add, ptr %gep.dst, align 4
  %gep.src.2 = getelementptr inbounds i32, ptr %dst.2.off, i64 %iv
  store i32 %lv, ptr %gep.src.2, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %exit, label %for.body

exit:
  ret void
}


; Final value is not guaranteed to be stored in an invariant address.
; We don't vectorize in that case.
;
; int sum = 0;
; for(i=0..N) {
;   int diff = y[i] - x[i];
;   if (diff > 0) {
;     sum = += diff;
;     *t = sum;
;   }
; }
; CHECK-LABEL: @reduc_cond_store
; CHECK-NOT: vector.body
define void @reduc_cond_store(ptr %t, ptr readonly %x, ptr readonly %y) {
entry:
  store i32 0, ptr %t, align 4
  br label %for.body

for.body:
  %sum = phi i32 [ 0, %entry ], [ %sum.2, %if.end ]
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %if.end ]
  %gep.y = getelementptr inbounds i32, ptr %y, i64 %iv
  %0 = load i32, ptr %gep.y, align 4
  %gep.x = getelementptr inbounds i32, ptr %x, i64 %iv
  %1 = load i32, ptr %gep.x, align 4
  %diff = sub nsw i32 %0, %1
  %cmp2 = icmp sgt i32 %diff, 0
  br i1 %cmp2, label %if.then, label %if.end

if.then:
  %sum.1 = add nsw i32 %diff, %sum
  store i32 %sum.1, ptr %t, align 4
  br label %if.end

if.end:
  %sum.2 = phi i32 [ %sum.1, %if.then ], [ %0, %for.body ]
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

; Check that we can vectorize code with several stores to an invariant address
; with condition that final reduction value is stored too.
;
;  int sum = 0;
;  for(int i=0; i < 1000; i+=2) {
;    sum += src[i];
;    dst[42] = sum;
;    sum += src[i+1];
;    dst[42] = sum;
;  }
; CHECK-LABEL: @reduc_store_inside_unrolled
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH:%.*]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY:%.*]] ]
; CHECK-NEXT:    [[VEC_IND:%.*]] = phi <4 x i64> [ <i64 0, i64 2, i64 4, i64 6>, [[VECTOR_PH]] ], [ [[VEC_IND_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_PHI:%.*]] = phi <4 x i32> [ zeroinitializer, [[VECTOR_PH]] ], [ [[TMP34:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[OFFSET_IDX:%.*]] = mul i64 [[INDEX]], 2
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[OFFSET_IDX]], 0
; CHECK-NEXT:    [[TMP1:%.*]] = add i64 [[OFFSET_IDX]], 2
; CHECK-NEXT:    [[TMP2:%.*]] = add i64 [[OFFSET_IDX]], 4
; CHECK-NEXT:    [[TMP3:%.*]] = add i64 [[OFFSET_IDX]], 6
; CHECK-NEXT:    [[TMP4:%.*]] = getelementptr inbounds i32, ptr [[SRC:%.*]], i64 [[TMP0]]
; CHECK-NEXT:    [[TMP5:%.*]] = getelementptr inbounds i32, ptr [[SRC]], i64 [[TMP1]]
; CHECK-NEXT:    [[TMP6:%.*]] = getelementptr inbounds i32, ptr [[SRC]], i64 [[TMP2]]
; CHECK-NEXT:    [[TMP7:%.*]] = getelementptr inbounds i32, ptr [[SRC]], i64 [[TMP3]]
; CHECK-NEXT:    [[TMP8:%.*]] = load i32, ptr [[TMP4]], align 4
; CHECK-NEXT:    [[TMP9:%.*]] = load i32, ptr [[TMP5]], align 4
; CHECK-NEXT:    [[TMP10:%.*]] = load i32, ptr [[TMP6]], align 4
; CHECK-NEXT:    [[TMP11:%.*]] = load i32, ptr [[TMP7]], align 4
; CHECK-NEXT:    [[TMP12:%.*]] = insertelement <4 x i32> poison, i32 [[TMP8]], i32 0
; CHECK-NEXT:    [[TMP13:%.*]] = insertelement <4 x i32> [[TMP12]], i32 [[TMP9]], i32 1
; CHECK-NEXT:    [[TMP14:%.*]] = insertelement <4 x i32> [[TMP13]], i32 [[TMP10]], i32 2
; CHECK-NEXT:    [[TMP15:%.*]] = insertelement <4 x i32> [[TMP14]], i32 [[TMP11]], i32 3
; CHECK-NEXT:    [[TMP16:%.*]] = add <4 x i32> [[TMP15]], [[VEC_PHI]]
; CHECK-NEXT:    [[TMP17:%.*]] = or disjoint <4 x i64> [[VEC_IND]], <i64 1, i64 1, i64 1, i64 1>
; CHECK-NEXT:    [[TMP18:%.*]] = extractelement <4 x i64> [[TMP17]], i32 0
; CHECK-NEXT:    [[TMP19:%.*]] = getelementptr inbounds i32, ptr [[SRC]], i64 [[TMP18]]
; CHECK-NEXT:    [[TMP20:%.*]] = extractelement <4 x i64> [[TMP17]], i32 1
; CHECK-NEXT:    [[TMP21:%.*]] = getelementptr inbounds i32, ptr [[SRC]], i64 [[TMP20]]
; CHECK-NEXT:    [[TMP22:%.*]] = extractelement <4 x i64> [[TMP17]], i32 2
; CHECK-NEXT:    [[TMP23:%.*]] = getelementptr inbounds i32, ptr [[SRC]], i64 [[TMP22]]
; CHECK-NEXT:    [[TMP24:%.*]] = extractelement <4 x i64> [[TMP17]], i32 3
; CHECK-NEXT:    [[TMP25:%.*]] = getelementptr inbounds i32, ptr [[SRC]], i64 [[TMP24]]
; CHECK-NEXT:    [[TMP26:%.*]] = load i32, ptr [[TMP19]], align 4
; CHECK-NEXT:    [[TMP27:%.*]] = load i32, ptr [[TMP21]], align 4
; CHECK-NEXT:    [[TMP28:%.*]] = load i32, ptr [[TMP23]], align 4
; CHECK-NEXT:    [[TMP29:%.*]] = load i32, ptr [[TMP25]], align 4
; CHECK-NEXT:    [[TMP30:%.*]] = insertelement <4 x i32> poison, i32 [[TMP26]], i32 0
; CHECK-NEXT:    [[TMP31:%.*]] = insertelement <4 x i32> [[TMP30]], i32 [[TMP27]], i32 1
; CHECK-NEXT:    [[TMP32:%.*]] = insertelement <4 x i32> [[TMP31]], i32 [[TMP28]], i32 2
; CHECK-NEXT:    [[TMP33:%.*]] = insertelement <4 x i32> [[TMP32]], i32 [[TMP29]], i32 3
; CHECK-NEXT:    [[TMP34]] = add <4 x i32> [[TMP33]], [[TMP16]]
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 4
; CHECK-NEXT:    [[VEC_IND_NEXT]] = add <4 x i64> [[VEC_IND]], <i64 8, i64 8, i64 8, i64 8>
; CHECK-NEXT:    [[TMP35:%.*]] = icmp eq i64 [[INDEX_NEXT]], 500
; CHECK-NEXT:    br i1 [[TMP35]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP14:![0-9]+]]
; CHECK:       middle.block:
; CHECK-NEXT:    [[TMP36:%.*]] = call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> [[TMP34]])
; CHECK-NEXT:    store i32 [[TMP36]], ptr [[GEP_DST:%.*]], align 4
; CHECK-NEXT:    br i1 true, label [[EXIT:%.*]], label [[SCALAR_PH:%.*]]
define void @reduc_store_inside_unrolled(ptr %dst, ptr readonly %src) {
entry:
  %gep.dst = getelementptr inbounds i32, ptr %dst, i64 42
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %sum.2, %for.body ]
  %gep.src = getelementptr inbounds i32, ptr %src, i64 %iv
  %0 = load i32, ptr %gep.src, align 4
  %sum.1 = add nsw i32 %0, %sum
  store i32 %sum.1, ptr %gep.dst, align 4
  %1 = or disjoint i64 %iv, 1
  %gep.src.1 = getelementptr inbounds i32, ptr %src, i64 %1
  %2 = load i32, ptr %gep.src.1, align 4
  %sum.2 = add nsw i32 %2, %sum.1
  store i32 %sum.2, ptr %gep.dst, align 4
  %iv.next = add nuw nsw i64 %iv, 2
  %cmp = icmp slt i64 %iv.next, 1000
  br i1 %cmp, label %for.body, label %exit

exit:
  ret void
}

; Check that we cannot vectorize code if stored value is not the final reduction
; value
;
;  int sum = 0;
;  for(int i=0; i < 1000; i++) {
;    sum += src[i];
;    dst[42] = sum + 1;
;  }
; CHECK-LABEL: @reduc_store_not_final_value
; CHECK-NOT: vector.body:
define void @reduc_store_not_final_value(ptr %dst, ptr readonly %src) {
entry:
  %gep.dst = getelementptr inbounds i32, ptr %dst, i64 42
  store i32 0, ptr %gep.dst, align 4
  br label %for.body

for.body:
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %gep.src = getelementptr inbounds i32, ptr %src, i64 %iv
  %0 = load i32, ptr %gep.src, align 4
  %add = add nsw i32 %sum, %0
  %sum_plus_one = add i32 %add, 1
  store i32 %sum_plus_one, ptr %gep.dst, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %exit, label %for.body

exit:
  ret void
}

; We cannot vectorize if two (or more) invariant stores exist in a loop.
;
;  int sum = 0;
;  for(int i=0; i < 1000; i+=2) {
;    sum += src[i];
;    dst[42] = sum;
;    sum += src[i+1];
;    other_dst[42] = sum;
;  }
; CHECK-LABEL: @reduc_double_invariant_store
; CHECK-NOT: vector.body:
define void @reduc_double_invariant_store(ptr %dst, ptr %other_dst, ptr readonly %src) {
entry:
  %gep.dst = getelementptr inbounds i32, ptr %dst, i64 42
  %gep.other_dst = getelementptr inbounds i32, ptr %other_dst, i64 42
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %sum.2, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %src, i64 %iv
  %0 = load i32, ptr %arrayidx, align 4
  %sum.1 = add nsw i32 %0, %sum
  store i32 %sum.1, ptr %gep.dst, align 4
  %1 = or disjoint i64 %iv, 1
  %arrayidx4 = getelementptr inbounds i32, ptr %src, i64 %1
  %2 = load i32, ptr %arrayidx4, align 4
  %sum.2 = add nsw i32 %2, %sum.1
  store i32 %sum.2, ptr %gep.other_dst, align 4
  %iv.next = add nuw nsw i64 %iv, 2
  %cmp = icmp slt i64 %iv.next, 1000
  br i1 %cmp, label %for.body, label %exit

exit:
  ret void
}

;  int sum = 0;
;  for(int i=0; i < 1000; i+=2) {
;    sum += src[i];
;    if (src[i+1] > 0)
;      dst[42] = sum;
;    sum += src[i+1];
;    dst[42] = sum;
;  }
; CHECK-LABEL: @reduc_store_middle_store_predicated
; CHECK: vector.body:
; CHECK-NOT: store i32 %{{[0-9]+}}, ptr %gep.dst
; CHECK: middle.block:
; CHECK-NEXT: [[TMP:%.*]] = call i32 @llvm.vector.reduce.add.v4i32
; CHECK-NEXT: store i32 [[TMP]], ptr %gep.dst
; CHECK: ret void
define void @reduc_store_middle_store_predicated(ptr %dst, ptr readonly %src) {
entry:
  %gep.dst = getelementptr inbounds i32, ptr %dst, i64 42
  br label %for.body

for.body:                                         ; preds = %latch, %entry
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %latch ]
  %sum = phi i32 [ 0, %entry ], [ %sum.2, %latch ]
  %gep.src = getelementptr inbounds i32, ptr %src, i64 %iv
  %0 = load i32, ptr %gep.src, align 4
  %sum.1 = add nsw i32 %0, %sum
  %cmp = icmp sgt i32 %0, 0
  br i1 %cmp, label %predicated, label %latch

predicated:                                       ; preds = %for.body
  store i32 %sum.1, ptr %gep.dst, align 4
  br label %latch

latch:                                            ; preds = %predicated, %for.body
  %1 = or disjoint i64 %iv, 1
  %gep.src.1 = getelementptr inbounds i32, ptr %src, i64 %1
  %2 = load i32, ptr %gep.src.1, align 4
  %sum.2 = add nsw i32 %2, %sum.1
  store i32 %sum.2, ptr %gep.dst, align 4
  %iv.next = add nuw nsw i64 %iv, 2
  %cmp.1 = icmp slt i64 %iv.next, 1000
  br i1 %cmp.1, label %for.body, label %exit

exit:                                 ; preds = %latch
  ret void
}

;  int sum = 0;
;  for(int i=0; i < 1000; i+=2) {
;    sum += src[i];
;    dst[42] = sum;
;    sum += src[i+1];
;    if (src[i+1] > 0)
;      dst[42] = sum;
;  }
; CHECK-LABEL: @reduc_store_final_store_predicated
; CHECK-NOT: vector.body:
define void @reduc_store_final_store_predicated(ptr %dst, ptr readonly %src) {
entry:
  %gep.dst = getelementptr inbounds i32, ptr %dst, i64 42
  br label %for.body

for.body:                                         ; preds = %latch, %entry
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %latch ]
  %sum = phi i32 [ 0, %entry ], [ %sum.1, %latch ]
  %arrayidx = getelementptr inbounds i32, ptr %src, i64 %iv
  %0 = load i32, ptr %arrayidx, align 4
  %sum.1 = add nsw i32 %0, %sum
  store i32 %sum.1, ptr %gep.dst, align 4
  %1 = or disjoint i64 %iv, 1
  %gep.src.1 = getelementptr inbounds i32, ptr %src, i64 %1
  %2 = load i32, ptr %gep.src.1, align 4
  %sum.2 = add nsw i32 %2, %sum.1
  %cmp1 = icmp sgt i32 %2, 0
  br i1 %cmp1, label %predicated, label %latch

predicated:                                       ; preds = %for.body
  store i32 %sum.2, ptr %gep.dst, align 4
  br label %latch

latch:                                            ; preds = %predicated, %for.body
  %iv.next = add nuw nsw i64 %iv, 2
  %cmp = icmp slt i64 %iv.next, 1000
  br i1 %cmp, label %for.body, label %exit

exit:                                 ; preds = %latch
  ret void
}

; Final reduction value is overwritten inside loop
;
; for(int i=0; i < 1000; i++) {
;   sum += src[i];
;   dst[42] = sum;
;   dst[42] = 0;
; }
; CHECK-LABEL: @reduc_store_final_store_overwritten
; CHECK-NOT: vector.body:
define void @reduc_store_final_store_overwritten(ptr %dst, ptr readonly %src) {
entry:
  %gep.dst = getelementptr inbounds i32, ptr %dst, i64 42
  br label %for.body

for.body:
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %gep.src = getelementptr inbounds i32, ptr %src, i64 %iv
  %0 = load i32, ptr %gep.src, align 4
  %add = add nsw i32 %sum, %0
  store i32 %add, ptr %gep.dst, align 4
  store i32 0, ptr %gep.dst, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %exit, label %for.body

exit:
  ret void
}

; Final value used outside of loop does not prevent vectorization
;
; int sum = 0;
; for(int i=0; i < 1000; i++) {
;   sum += src[i];
;   dst[42] = sum;
; }
; dst[43] = sum;
; CHECK-LABEL: @reduc_store_inoutside
; CHECK: vector.body:
; CHECK-NOT: store i32 %{{[0-9]+}}, ptr %gep.src
; CHECK: middle.block:
; CHECK-NEXT: [[TMP:%.*]] = call i32 @llvm.vector.reduce.add.v4i32
; CHECK-NEXT: store i32 [[TMP]], ptr %gep.dst
; CHECK: exit:
; CHECK: [[PHI:%.*]] = phi i32 [ [[TMP1:%.*]], %for.body ], [ [[TMP2:%.*]], %middle.block ]
; CHECK: [[ADDR:%.*]] = getelementptr inbounds i32, ptr %dst, i64 43
; CHECK: store i32 [[PHI]], ptr [[ADDR]]
; CHECK: ret void
define void @reduc_store_inoutside(ptr %dst, ptr readonly %src) {
entry:
  %gep.dst = getelementptr inbounds i32, ptr %dst, i64 42
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %sum.1, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %src, i64 %iv
  %0 = load i32, ptr %arrayidx, align 4
  %sum.1 = add nsw i32 %0, %sum
  store i32 %sum.1, ptr %gep.dst, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %exit, label %for.body

exit:
  %sum.lcssa = phi i32 [ %sum.1, %for.body ]
  %gep.dst.1 = getelementptr inbounds i32, ptr %dst, i64 43
  store i32 %sum.lcssa, ptr %gep.dst.1, align 4
  ret void
}

; Test for PR55540.
define void @test_drop_poison_generating_dead_recipe(ptr %dst) {
; CHECK-LABEL: @test_drop_poison_generating_dead_recipe(
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i32 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[VEC_PHI:%.*]] = phi <4 x i64> [ zeroinitializer, %vector.ph ], [ [[TMP0:%.*]], %vector.body ]
; CHECK-NEXT:    [[TMP0]] = add <4 x i64> [[VEC_PHI]], <i64 -31364, i64 -31364, i64 -31364, i64 -31364>
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i32 [[INDEX]], 4
; CHECK-NEXT:    [[TMP1:%.*]] = icmp eq i32 [[INDEX_NEXT]], 360
; CHECK-NEXT:    br i1 [[TMP1]], label %middle.block, label %vector.body
; CHECK:       middle.block:
; CHECK-NEXT:    [[TMP2:%.*]] = call i64 @llvm.vector.reduce.add.v4i64(<4 x i64> [[TMP0]])
; CHECK-NEXT:    store i64 [[TMP2]], ptr [[DST:%.*]], align 8
; CHECK-NEXT:    br i1 false, label %exit, label %scalar.ph
; CHECK:       scalar.ph:
;
entry:
  br label %body

body:
  %red = phi i64 [ 0, %entry ], [ %red.next, %body ]
  %iv = phi i32 [ 2, %entry ], [ %iv.next, %body ]
  %add.1 = add nuw i64 %red, -23523
  store i64 %add.1, ptr %dst, align 8
  %red.next = add nuw i64 %red, -31364
  store i64 %red.next, ptr %dst, align 8
  %iv.next = add nuw nsw i32 %iv, 1
  %ec = icmp ugt i32 %iv, 363
  br i1 %ec, label %exit, label %body

exit:
  ret void
}

define void @reduc_store_invariant_addr_not_hoisted(ptr %dst, ptr readonly %src) {
; CHECK-LABEL: @reduc_store_invariant_addr_not_hoisted
; CHECK-NOT: vector.body:
entry:
  br label %for.body

for.body:
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %gep.src = getelementptr inbounds i32, ptr %src, i64 %iv
  %0 = load i32, ptr %gep.src, align 4
  %add = add nsw i32 %sum, %0
  %gep.dst = getelementptr inbounds i32, ptr %dst, i64 42
  store i32 %add, ptr %gep.dst, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %exit, label %for.body

exit:
  ret void
}

; Make sure we can vectorize loop with a non-reduction value stored to an
; invariant address that is calculated inside loop.
define i32 @non_reduc_store_invariant_addr_not_hoisted(ptr %dst, ptr readonly %src) {
; CHECK-LABEL: @non_reduc_store_invariant_addr_not_hoisted
; CHECK: vector.body:
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %gep.src = getelementptr inbounds i32, ptr %src, i64 %iv
  %0 = load i32, ptr %gep.src, align 4
  %add = add nsw i32 %sum, %0
  %gep.dst = getelementptr inbounds i32, ptr %dst, i64 42
  store i32 0, ptr %gep.dst, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %exit, label %for.body

exit:                                             ; preds = %for.body
  %add.lcssa = phi i32 [ %add, %for.body ]
  ret i32 %add.lcssa
}

; Make sure that if there are several reductions in the loop, the order of invariant stores sank outside of the loop is preserved
; See https://github.com/llvm/llvm-project/issues/64047
define void @reduc_add_mul_store_same_ptr(ptr %dst, ptr readonly %src) {
; CHECK-LABEL: define void @reduc_add_mul_store_same_ptr
; CHECK:       middle.block:
; CHECK-NEXT:    [[TMP6:%.*]] = call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> [[TMP3:%.*]])
; CHECK-NEXT:    [[TMP7:%.*]] = call i32 @llvm.vector.reduce.mul.v4i32(<4 x i32> [[TMP4:%.*]])
; CHECK-NEXT:    store i32 [[TMP6]], ptr %dst, align 4
; CHECK-NEXT:    store i32 [[TMP7]], ptr %dst, align 4
;
entry:
  br label %for.body

for.body:
  %sum = phi i32 [ 0, %entry ], [ %sum.next, %for.body ]
  %mul = phi i32 [ 1, %entry ], [ %mul.next, %for.body ]
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %gep.src = getelementptr inbounds i32, ptr %src, i64 %iv
  %0 = load i32, ptr %gep.src, align 4
  %sum.next = add nsw i32 %sum, %0
  store i32 %sum.next, ptr %dst, align 4
  %mul.next = mul nsw i32 %mul, %0
  store i32 %mul.next, ptr %dst, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %exit, label %for.body

exit:
  ret void
}

define void @reduc_mul_add_store_same_ptr(ptr %dst, ptr readonly %src) {
; CHECK-LABEL: define void @reduc_mul_add_store_same_ptr
; CHECK:       middle.block:
; CHECK-NEXT:    [[TMP6:%.*]] = call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> [[TMP4:%.*]])
; CHECK-NEXT:    [[TMP7:%.*]] = call i32 @llvm.vector.reduce.mul.v4i32(<4 x i32> [[TMP3:%.*]])
; CHECK-NEXT:    store i32 [[TMP7]], ptr %dst, align 4
; CHECK-NEXT:    store i32 [[TMP6]], ptr %dst, align 4
;
entry:
  br label %for.body

for.body:
  %sum = phi i32 [ 0, %entry ], [ %sum.next, %for.body ]
  %mul = phi i32 [ 1, %entry ], [ %mul.next, %for.body ]
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %gep.src = getelementptr inbounds i32, ptr %src, i64 %iv
  %0 = load i32, ptr %gep.src, align 4
  %mul.next = mul nsw i32 %mul, %0
  store i32 %mul.next, ptr %dst, align 4
  %sum.next = add nsw i32 %sum, %0
  store i32 %sum.next, ptr %dst, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %exit, label %for.body

exit:
  ret void
}

; Same as above but storing is done to two different pointers and they can be aliased
define void @reduc_add_mul_store_different_ptr(ptr %dst1, ptr %dst2, ptr readonly %src) {
; CHECK-LABEL: define void @reduc_add_mul_store_different_ptr
; CHECK:       middle.block:
; CHECK-NEXT:    [[TMP6:%.*]] = call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> [[TMP3:%.*]])
; CHECK-NEXT:    [[TMP7:%.*]] = call i32 @llvm.vector.reduce.mul.v4i32(<4 x i32> [[TMP4:%.*]])
; CHECK-NEXT:    store i32 [[TMP6]], ptr %dst1, align 4
; CHECK-NEXT:    store i32 [[TMP7]], ptr %dst2, align 4
;
entry:
  br label %for.body

for.body:
  %sum = phi i32 [ 0, %entry ], [ %sum.next, %for.body ]
  %mul = phi i32 [ 1, %entry ], [ %mul.next, %for.body ]
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %gep.src = getelementptr inbounds i32, ptr %src, i64 %iv
  %0 = load i32, ptr %gep.src, align 4
  %sum.next = add nsw i32 %sum, %0
  store i32 %sum.next, ptr %dst1, align 4
  %mul.next = mul nsw i32 %mul, %0
  store i32 %mul.next, ptr %dst2, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %exit, label %for.body

exit:
  ret void
}

define void @reduc_mul_add_store_different_ptr(ptr %dst1, ptr %dst2, ptr readonly %src) {
; CHECK-LABEL: define void @reduc_mul_add_store_different_ptr
; CHECK:       middle.block:
; CHECK-NEXT:    [[TMP6:%.*]] = call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> [[TMP4:%.*]])
; CHECK-NEXT:    [[TMP7:%.*]] = call i32 @llvm.vector.reduce.mul.v4i32(<4 x i32> [[TMP3:%.*]])
; CHECK-NEXT:    store i32 [[TMP7]], ptr %dst1, align 4
; CHECK-NEXT:    store i32 [[TMP6]], ptr %dst2, align 4
;
entry:
  br label %for.body

for.body:
  %sum = phi i32 [ 0, %entry ], [ %sum.next, %for.body ]
  %mul = phi i32 [ 1, %entry ], [ %mul.next, %for.body ]
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %gep.src = getelementptr inbounds i32, ptr %src, i64 %iv
  %0 = load i32, ptr %gep.src, align 4
  %mul.next = mul nsw i32 %mul, %0
  store i32 %mul.next, ptr %dst1, align 4
  %sum.next = add nsw i32 %sum, %0
  store i32 %sum.next, ptr %dst2, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %exit, label %for.body

exit:
  ret void
}
