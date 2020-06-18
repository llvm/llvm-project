; RUN: opt -S -loop-vectorize -instcombine -force-vector-width=4 -force-vector-interleave=1 -enable-interleaved-mem-accesses=true -runtime-memory-check-threshold=24 < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"

; Check vectorization on an interleaved load group of factor 2 and an interleaved
; store group of factor 2.

; int AB[1024];
; int CD[1024];
;  void test_array_load2_store2(int C, int D) {
;   for (int i = 0; i < 1024; i+=2) {
;     int A = AB[i];
;     int B = AB[i+1];
;     CD[i] = A + C;
;     CD[i+1] = B * D;
;   }
; }

; CHECK-LABEL: @test_array_load2_store2(
; CHECK: %wide.vec = load <8 x i32>, <8 x i32>* %{{.*}}, align 4
; CHECK: shufflevector <8 x i32> %wide.vec, <8 x i32> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
; CHECK: shufflevector <8 x i32> %wide.vec, <8 x i32> undef, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
; CHECK: add nsw <4 x i32>
; CHECK: mul nsw <4 x i32>
; CHECK: %interleaved.vec = shufflevector <4 x i32> {{.*}}, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
; CHECK: store <8 x i32> %interleaved.vec, <8 x i32>* %{{.*}}, align 4

@AB = common global [1024 x i32] zeroinitializer, align 4
@CD = common global [1024 x i32] zeroinitializer, align 4

define void @test_array_load2_store2(i32 %C, i32 %D) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx0 = getelementptr inbounds [1024 x i32], [1024 x i32]* @AB, i64 0, i64 %indvars.iv
  %tmp = load i32, i32* %arrayidx0, align 4
  %tmp1 = or i64 %indvars.iv, 1
  %arrayidx1 = getelementptr inbounds [1024 x i32], [1024 x i32]* @AB, i64 0, i64 %tmp1
  %tmp2 = load i32, i32* %arrayidx1, align 4
  %add = add nsw i32 %tmp, %C
  %mul = mul nsw i32 %tmp2, %D
  %arrayidx2 = getelementptr inbounds [1024 x i32], [1024 x i32]* @CD, i64 0, i64 %indvars.iv
  store i32 %add, i32* %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds [1024 x i32], [1024 x i32]* @CD, i64 0, i64 %tmp1
  store i32 %mul, i32* %arrayidx3, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 2
  %cmp = icmp slt i64 %indvars.iv.next, 1024
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}

; int A[3072];
; struct ST S[1024];
; void test_struct_st3() {
;   int *ptr = A;
;   for (int i = 0; i < 1024; i++) {
;     int X1 = *ptr++;
;     int X2 = *ptr++;
;     int X3 = *ptr++;
;     T[i].x = X1 + 1;
;     T[i].y = X2 + 2;
;     T[i].z = X3 + 3;
;   }
; }

; CHECK-LABEL: @test_struct_array_load3_store3(
; CHECK: %wide.vec = load <12 x i32>, <12 x i32>* {{.*}}, align 4
; CHECK: shufflevector <12 x i32> %wide.vec, <12 x i32> undef, <4 x i32> <i32 0, i32 3, i32 6, i32 9>
; CHECK: shufflevector <12 x i32> %wide.vec, <12 x i32> undef, <4 x i32> <i32 1, i32 4, i32 7, i32 10>
; CHECK: shufflevector <12 x i32> %wide.vec, <12 x i32> undef, <4 x i32> <i32 2, i32 5, i32 8, i32 11>
; CHECK: add nsw <4 x i32> {{.*}}, <i32 1, i32 1, i32 1, i32 1>
; CHECK: add nsw <4 x i32> {{.*}}, <i32 2, i32 2, i32 2, i32 2>
; CHECK: add nsw <4 x i32> {{.*}}, <i32 3, i32 3, i32 3, i32 3>
; CHECK: shufflevector <4 x i32> {{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; CHECK: shufflevector <4 x i32> {{.*}}, <4 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
; CHECK: %interleaved.vec = shufflevector <8 x i32> {{.*}}, <12 x i32> <i32 0, i32 4, i32 8, i32 1, i32 5, i32 9, i32 2, i32 6, i32 10, i32 3, i32 7, i32 11>
; CHECK: store <12 x i32> %interleaved.vec, <12 x i32>* {{.*}}, align 4

%struct.ST3 = type { i32, i32, i32 }
@A = common global [3072 x i32] zeroinitializer, align 4
@S = common global [1024 x %struct.ST3] zeroinitializer, align 4

define void @test_struct_array_load3_store3() {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %ptr.016 = phi i32* [ getelementptr inbounds ([3072 x i32], [3072 x i32]* @A, i64 0, i64 0), %entry ], [ %incdec.ptr2, %for.body ]
  %incdec.ptr = getelementptr inbounds i32, i32* %ptr.016, i64 1
  %tmp = load i32, i32* %ptr.016, align 4
  %incdec.ptr1 = getelementptr inbounds i32, i32* %ptr.016, i64 2
  %tmp1 = load i32, i32* %incdec.ptr, align 4
  %incdec.ptr2 = getelementptr inbounds i32, i32* %ptr.016, i64 3
  %tmp2 = load i32, i32* %incdec.ptr1, align 4
  %add = add nsw i32 %tmp, 1
  %x = getelementptr inbounds [1024 x %struct.ST3], [1024 x %struct.ST3]* @S, i64 0, i64 %indvars.iv, i32 0
  store i32 %add, i32* %x, align 4
  %add3 = add nsw i32 %tmp1, 2
  %y = getelementptr inbounds [1024 x %struct.ST3], [1024 x %struct.ST3]* @S, i64 0, i64 %indvars.iv, i32 1
  store i32 %add3, i32* %y, align 4
  %add6 = add nsw i32 %tmp2, 3
  %z = getelementptr inbounds [1024 x %struct.ST3], [1024 x %struct.ST3]* @S, i64 0, i64 %indvars.iv, i32 2
  store i32 %add6, i32* %z, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

; Check vectorization on an interleaved load group of factor 4.

; struct ST4{
;   int x;
;   int y;
;   int z;
;   int w;
; };
; int test_struct_load4(struct ST4 *S) {
;   int r = 0;
;   for (int i = 0; i < 1024; i++) {
;      r += S[i].x;
;      r -= S[i].y;
;      r += S[i].z;
;      r -= S[i].w;
;   }
;   return r;
; }

%struct.ST4 = type { i32, i32, i32, i32 }

define i32 @test_struct_load4(%struct.ST4* nocapture readonly %S) {
; CHECK-LABEL: @test_struct_load4(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 false, label [[SCALAR_PH:%.*]], label [[VECTOR_PH:%.*]]
; CHECK:       vector.ph:
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_PHI:%.*]] = phi <4 x i32> [ zeroinitializer, [[VECTOR_PH]] ], [ [[TMP5:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP0:%.*]] = getelementptr inbounds [[STRUCT_ST4:%.*]], %struct.ST4* [[S:%.*]], i64 [[INDEX]], i32 0
; CHECK-NEXT:    [[TMP1:%.*]] = bitcast i32* [[TMP0]] to <16 x i32>*
; CHECK-NEXT:    [[WIDE_VEC:%.*]] = load <16 x i32>, <16 x i32>* [[TMP1]], align 4
; CHECK-NEXT:    [[STRIDED_VEC:%.*]] = shufflevector <16 x i32> [[WIDE_VEC]], <16 x i32> undef, <4 x i32> <i32 0, i32 4, i32 8, i32 12>
; CHECK-NEXT:    [[STRIDED_VEC1:%.*]] = shufflevector <16 x i32> [[WIDE_VEC]], <16 x i32> undef, <4 x i32> <i32 1, i32 5, i32 9, i32 13>
; CHECK-NEXT:    [[STRIDED_VEC2:%.*]] = shufflevector <16 x i32> [[WIDE_VEC]], <16 x i32> undef, <4 x i32> <i32 2, i32 6, i32 10, i32 14>
; CHECK-NEXT:    [[STRIDED_VEC3:%.*]] = shufflevector <16 x i32> [[WIDE_VEC]], <16 x i32> undef, <4 x i32> <i32 3, i32 7, i32 11, i32 15>
; CHECK-NEXT:    [[TMP2:%.*]] = add <4 x i32> [[STRIDED_VEC]], [[VEC_PHI]]
; CHECK-NEXT:    [[TMP3:%.*]] = add <4 x i32> [[TMP2]], [[STRIDED_VEC2]]
; CHECK-NEXT:    [[TMP4:%.*]] = add <4 x i32> [[STRIDED_VEC1]], [[STRIDED_VEC3]]
; CHECK-NEXT:    [[TMP5]] = sub <4 x i32> [[TMP3]], [[TMP4]]
; CHECK-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], 4
; CHECK-NEXT:    [[TMP6:%.*]] = icmp eq i64 [[INDEX_NEXT]], 1024
; CHECK-NEXT:    br i1 [[TMP6]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop !6
; CHECK:       middle.block:
; CHECK-NEXT:    [[RDX_SHUF:%.*]] = shufflevector <4 x i32> [[TMP5]], <4 x i32> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
; CHECK-NEXT:    [[BIN_RDX:%.*]] = add <4 x i32> [[TMP5]], [[RDX_SHUF]]
; CHECK-NEXT:    [[RDX_SHUF4:%.*]] = shufflevector <4 x i32> [[BIN_RDX]], <4 x i32> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
; CHECK-NEXT:    [[BIN_RDX5:%.*]] = add <4 x i32> [[BIN_RDX]], [[RDX_SHUF4]]
; CHECK-NEXT:    [[TMP7:%.*]] = extractelement <4 x i32> [[BIN_RDX5]], i32 0
; CHECK-NEXT:    br i1 true, label [[FOR_END:%.*]], label [[SCALAR_PH]]
; CHECK:       scalar.ph:
; CHECK-NEXT:    br label [[FOR_BODY:%.*]]
; CHECK:       for.body:
; CHECK-NEXT:    br i1 undef, label [[FOR_END]], label [[FOR_BODY]], !llvm.loop !7
; CHECK:       for.end:
; CHECK-NEXT:    [[SUB8_LCSSA:%.*]] = phi i32 [ undef, [[FOR_BODY]] ], [ [[TMP7]], [[MIDDLE_BLOCK]] ]
; CHECK-NEXT:    ret i32 [[SUB8_LCSSA]]
;
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %r.022 = phi i32 [ 0, %entry ], [ %sub8, %for.body ]
  %x = getelementptr inbounds %struct.ST4, %struct.ST4* %S, i64 %indvars.iv, i32 0
  %tmp = load i32, i32* %x, align 4
  %add = add nsw i32 %tmp, %r.022
  %y = getelementptr inbounds %struct.ST4, %struct.ST4* %S, i64 %indvars.iv, i32 1
  %tmp1 = load i32, i32* %y, align 4
  %sub = sub i32 %add, %tmp1
  %z = getelementptr inbounds %struct.ST4, %struct.ST4* %S, i64 %indvars.iv, i32 2
  %tmp2 = load i32, i32* %z, align 4
  %add5 = add nsw i32 %sub, %tmp2
  %w = getelementptr inbounds %struct.ST4, %struct.ST4* %S, i64 %indvars.iv, i32 3
  %tmp3 = load i32, i32* %w, align 4
  %sub8 = sub i32 %add5, %tmp3
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret i32 %sub8
}

; Check vectorization on an interleaved store group of factor 4.

; void test_struct_store4(int *A, struct ST4 *B) {
;   int *ptr = A;
;   for (int i = 0; i < 1024; i++) {
;     int X = *ptr++;
;     B[i].x = X + 1;
;     B[i].y = X * 2;
;     B[i].z = X + 3;
;     B[i].w = X + 4;
;   }
; }

; CHECK-LABEL: @test_struct_store4(
; CHECK: %[[LD:.*]] = load <4 x i32>, <4 x i32>*
; CHECK: add nsw <4 x i32> %[[LD]], <i32 1, i32 1, i32 1, i32 1>
; CHECK: shl nsw <4 x i32> %[[LD]], <i32 1, i32 1, i32 1, i32 1>
; CHECK: add nsw <4 x i32> %[[LD]], <i32 3, i32 3, i32 3, i32 3>
; CHECK: add nsw <4 x i32> %[[LD]], <i32 4, i32 4, i32 4, i32 4>
; CHECK: shufflevector <4 x i32> {{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; CHECK: shufflevector <4 x i32> {{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; CHECK: %interleaved.vec = shufflevector <8 x i32> {{.*}}, <16 x i32> <i32 0, i32 4, i32 8, i32 12, i32 1, i32 5, i32 9, i32 13, i32 2, i32 6, i32 10, i32 14, i32 3, i32 7, i32 11, i32 15>
; CHECK: store <16 x i32> %interleaved.vec, <16 x i32>* {{.*}}, align 4

define void @test_struct_store4(i32* noalias nocapture readonly %A, %struct.ST4* noalias nocapture %B) {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %ptr.024 = phi i32* [ %A, %entry ], [ %incdec.ptr, %for.body ]
  %incdec.ptr = getelementptr inbounds i32, i32* %ptr.024, i64 1
  %tmp = load i32, i32* %ptr.024, align 4
  %add = add nsw i32 %tmp, 1
  %x = getelementptr inbounds %struct.ST4, %struct.ST4* %B, i64 %indvars.iv, i32 0
  store i32 %add, i32* %x, align 4
  %mul = shl nsw i32 %tmp, 1
  %y = getelementptr inbounds %struct.ST4, %struct.ST4* %B, i64 %indvars.iv, i32 1
  store i32 %mul, i32* %y, align 4
  %add3 = add nsw i32 %tmp, 3
  %z = getelementptr inbounds %struct.ST4, %struct.ST4* %B, i64 %indvars.iv, i32 2
  store i32 %add3, i32* %z, align 4
  %add6 = add nsw i32 %tmp, 4
  %w = getelementptr inbounds %struct.ST4, %struct.ST4* %B, i64 %indvars.iv, i32 3
  store i32 %add6, i32* %w, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; Check vectorization on a reverse interleaved load group of factor 2 and
; a reverse interleaved store group of factor 2.

; struct ST2 {
;  int x;
;  int y;
; };
;
; void test_reversed_load2_store2(struct ST2 *A, struct ST2 *B) {
;   for (int i = 1023; i >= 0; i--) {
;     int a = A[i].x + i;  // interleaved load of index 0
;     int b = A[i].y - i;  // interleaved load of index 1
;     B[i].x = a;          // interleaved store of index 0
;     B[i].y = b;          // interleaved store of index 1
;   }
; }

; CHECK-LABEL: @test_reversed_load2_store2(
; CHECK: %[[G0:.+]] = getelementptr inbounds %struct.ST2, %struct.ST2* %A, i64 %offset.idx, i32 0
; CHECK: %[[G1:.+]] = getelementptr inbounds i32, i32* %[[G0]], i64 -6
; CHECK: %[[B0:.+]] = bitcast i32* %[[G1]] to <8 x i32>*
; CHECK: %wide.vec = load <8 x i32>, <8 x i32>* %[[B0]], align 4
; CHECK: shufflevector <8 x i32> %wide.vec, <8 x i32> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
; CHECK: shufflevector <4 x i32> {{.*}}, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
; CHECK: shufflevector <8 x i32> %wide.vec, <8 x i32> undef, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
; CHECK: shufflevector <4 x i32> {{.*}}, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
; CHECK: add nsw <4 x i32>
; CHECK: sub nsw <4 x i32>
; CHECK: %[[G2:.+]] = getelementptr inbounds %struct.ST2, %struct.ST2* %B, i64 %offset.idx, i32 1
; CHECK: %[[G3:.+]] = getelementptr inbounds i32, i32* %[[G2]], i64 -7
; CHECK: %[[B1:.+]] = bitcast i32* %[[G3]] to <8 x i32>*
; CHECK: shufflevector <4 x i32> {{.*}}, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
; CHECK: shufflevector <4 x i32> {{.*}}, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
; CHECK: %interleaved.vec = shufflevector <4 x i32> {{.*}}, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
; CHECK: store <8 x i32> %interleaved.vec, <8 x i32>* %[[B1]], align 4

%struct.ST2 = type { i32, i32 }

define void @test_reversed_load2_store2(%struct.ST2* noalias nocapture readonly %A, %struct.ST2* noalias nocapture %B) {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 1023, %entry ], [ %indvars.iv.next, %for.body ]
  %x = getelementptr inbounds %struct.ST2, %struct.ST2* %A, i64 %indvars.iv, i32 0
  %tmp = load i32, i32* %x, align 4
  %tmp1 = trunc i64 %indvars.iv to i32
  %add = add nsw i32 %tmp, %tmp1
  %y = getelementptr inbounds %struct.ST2, %struct.ST2* %A, i64 %indvars.iv, i32 1
  %tmp2 = load i32, i32* %y, align 4
  %sub = sub nsw i32 %tmp2, %tmp1
  %x5 = getelementptr inbounds %struct.ST2, %struct.ST2* %B, i64 %indvars.iv, i32 0
  store i32 %add, i32* %x5, align 4
  %y8 = getelementptr inbounds %struct.ST2, %struct.ST2* %B, i64 %indvars.iv, i32 1
  store i32 %sub, i32* %y8, align 4
  %indvars.iv.next = add nsw i64 %indvars.iv, -1
  %cmp = icmp sgt i64 %indvars.iv, 0
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}

; Check vectorization on an interleaved load group of factor 2 with 1 gap
; (missing the load of odd elements). Because the vectorized loop would
; speculatively access memory out-of-bounds, we must execute at least one
; iteration of the scalar loop.

; void even_load_static_tc(int *A, int *B) {
;  for (unsigned i = 0; i < 1024; i+=2)
;     B[i/2] = A[i] * 2;
; }

; CHECK-LABEL: @even_load_static_tc(
; CHECK: vector.body:
; CHECK:   %wide.vec = load <8 x i32>, <8 x i32>* %{{.*}}, align 4
; CHECK:   %strided.vec = shufflevector <8 x i32> %wide.vec, <8 x i32> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
; CHECK:   icmp eq i64 %index.next, 508
; CHECK: middle.block:
; CHECK:   br i1 false, label %for.cond.cleanup, label %scalar.ph

define void @even_load_static_tc(i32* noalias nocapture readonly %A, i32* noalias nocapture %B) {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp = load i32, i32* %arrayidx, align 4
  %mul = shl nsw i32 %tmp, 1
  %tmp1 = lshr exact i64 %indvars.iv, 1
  %arrayidx2 = getelementptr inbounds i32, i32* %B, i64 %tmp1
  store i32 %mul, i32* %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 2
  %cmp = icmp ult i64 %indvars.iv.next, 1024
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}

; Check vectorization on an interleaved load group of factor 2 with 1 gap
; (missing the load of odd elements). Because the vectorized loop would
; speculatively access memory out-of-bounds, we must execute at least one
; iteration of the scalar loop.

; void even_load_dynamic_tc(int *A, int *B, unsigned N) {
;  for (unsigned i = 0; i < N; i+=2)
;     B[i/2] = A[i] * 2;
; }

; CHECK-LABEL: @even_load_dynamic_tc(
; CHECK: vector.ph:
; CHECK:   %n.mod.vf = and i64 %[[N:[a-zA-Z0-9]+]], 3
; CHECK:   %[[IsZero:[a-zA-Z0-9]+]] = icmp eq i64 %n.mod.vf, 0
; CHECK:   %[[R:[a-zA-Z0-9]+]] = select i1 %[[IsZero]], i64 4, i64 %n.mod.vf
; CHECK:   %n.vec = sub i64 %[[N]], %[[R]]
; CHECK: vector.body:
; CHECK:   %wide.vec = load <8 x i32>, <8 x i32>* %{{.*}}, align 4
; CHECK:   %strided.vec = shufflevector <8 x i32> %wide.vec, <8 x i32> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
; CHECK:   icmp eq i64 %index.next, %n.vec
; CHECK: middle.block:
; CHECK:   br i1 false, label %for.cond.cleanup, label %scalar.ph

define void @even_load_dynamic_tc(i32* noalias nocapture readonly %A, i32* noalias nocapture %B, i64 %N) {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp = load i32, i32* %arrayidx, align 4
  %mul = shl nsw i32 %tmp, 1
  %tmp1 = lshr exact i64 %indvars.iv, 1
  %arrayidx2 = getelementptr inbounds i32, i32* %B, i64 %tmp1
  store i32 %mul, i32* %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 2
  %cmp = icmp ult i64 %indvars.iv.next, %N
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}

; Check vectorization on a reverse interleaved load group of factor 2 with 1
; gap and a reverse interleaved store group of factor 2. The interleaved load
; group should be removed since it has a gap and is reverse.

; struct pair {
;  int x;
;  int y;
; };
;
; void load_gap_reverse(struct pair *P1, struct pair *P2, int X) {
;   for (int i = 1023; i >= 0; i--) {
;     int a = X + i;
;     int b = A[i].y - i;
;     B[i].x = a;
;     B[i].y = b;
;   }
; }

; CHECK-LABEL: @load_gap_reverse(
; CHECK-NOT: %wide.vec = load <8 x i64>, <8 x i64>* %{{.*}}, align 8
; CHECK-NOT: %strided.vec = shufflevector <8 x i64> %wide.vec, <8 x i64> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>

%pair = type { i64, i64 }
define void @load_gap_reverse(%pair* noalias nocapture readonly %P1, %pair* noalias nocapture readonly %P2, i64 %X) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 1023, %entry ], [ %i.next, %for.body ]
  %0 = add nsw i64 %X, %i
  %1 = getelementptr inbounds %pair, %pair* %P1, i64 %i, i32 0
  %2 = getelementptr inbounds %pair, %pair* %P2, i64 %i, i32 1
  %3 = load i64, i64* %2, align 8
  %4 = sub nsw i64 %3, %i
  store i64 %0, i64* %1, align 8
  store i64 %4, i64* %2, align 8
  %i.next = add nsw i64 %i, -1
  %cond = icmp sgt i64 %i, 0
  br i1 %cond, label %for.body, label %for.exit

for.exit:
  ret void
}

; Check vectorization on interleaved access groups identified from mixed
; loads/stores.
; void mixed_load2_store2(int *A, int *B) {
;   for (unsigned i = 0; i < 1024; i+=2)  {
;     B[i] = A[i] * A[i+1];
;     B[i+1] = A[i] + A[i+1];
;   }
; }

; CHECK-LABEL: @mixed_load2_store2(
; CHECK: %wide.vec = load <8 x i32>, <8 x i32>* {{.*}}, align 4
; CHECK: shufflevector <8 x i32> %wide.vec, <8 x i32> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
; CHECK: shufflevector <8 x i32> %wide.vec, <8 x i32> undef, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
; CHECK: %interleaved.vec = shufflevector <4 x i32> %{{.*}}, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
; CHECK: store <8 x i32> %interleaved.vec

define void @mixed_load2_store2(i32* noalias nocapture readonly %A, i32* noalias nocapture %B) {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp = load i32, i32* %arrayidx, align 4
  %tmp1 = or i64 %indvars.iv, 1
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i64 %tmp1
  %tmp2 = load i32, i32* %arrayidx2, align 4
  %mul = mul nsw i32 %tmp2, %tmp
  %arrayidx4 = getelementptr inbounds i32, i32* %B, i64 %indvars.iv
  store i32 %mul, i32* %arrayidx4, align 4
  %tmp3 = load i32, i32* %arrayidx, align 4
  %tmp4 = load i32, i32* %arrayidx2, align 4
  %add10 = add nsw i32 %tmp4, %tmp3
  %arrayidx13 = getelementptr inbounds i32, i32* %B, i64 %tmp1
  store i32 %add10, i32* %arrayidx13, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 2
  %cmp = icmp ult i64 %indvars.iv.next, 1024
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}

; Check vectorization on interleaved access groups identified from mixed
; loads/stores.
; void mixed_load3_store3(int *A) {
;   for (unsigned i = 0; i < 1024; i++)  {
;     *A++ += i;
;     *A++ += i;
;     *A++ += i;
;   }
; }

; CHECK-LABEL: @mixed_load3_store3(
; CHECK: %wide.vec = load <12 x i32>, <12 x i32>* {{.*}}, align 4
; CHECK: shufflevector <12 x i32> %wide.vec, <12 x i32> undef, <4 x i32> <i32 0, i32 3, i32 6, i32 9>
; CHECK: shufflevector <12 x i32> %wide.vec, <12 x i32> undef, <4 x i32> <i32 1, i32 4, i32 7, i32 10>
; CHECK: shufflevector <12 x i32> %wide.vec, <12 x i32> undef, <4 x i32> <i32 2, i32 5, i32 8, i32 11>
; CHECK: %interleaved.vec = shufflevector <8 x i32> %{{.*}}, <12 x i32> <i32 0, i32 4, i32 8, i32 1, i32 5, i32 9, i32 2, i32 6, i32 10, i32 3, i32 7, i32 11>
; CHECK: store <12 x i32> %interleaved.vec, <12 x i32>* %{{.*}}, align 4

define void @mixed_load3_store3(i32* nocapture %A) {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %for.body, %entry
  %i.013 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %A.addr.012 = phi i32* [ %A, %entry ], [ %incdec.ptr3, %for.body ]
  %incdec.ptr = getelementptr inbounds i32, i32* %A.addr.012, i64 1
  %tmp = load i32, i32* %A.addr.012, align 4
  %add = add i32 %tmp, %i.013
  store i32 %add, i32* %A.addr.012, align 4
  %incdec.ptr1 = getelementptr inbounds i32, i32* %A.addr.012, i64 2
  %tmp1 = load i32, i32* %incdec.ptr, align 4
  %add2 = add i32 %tmp1, %i.013
  store i32 %add2, i32* %incdec.ptr, align 4
  %incdec.ptr3 = getelementptr inbounds i32, i32* %A.addr.012, i64 3
  %tmp2 = load i32, i32* %incdec.ptr1, align 4
  %add4 = add i32 %tmp2, %i.013
  store i32 %add4, i32* %incdec.ptr1, align 4
  %inc = add nuw nsw i32 %i.013, 1
  %exitcond = icmp eq i32 %inc, 1024
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; Check vectorization on interleaved access groups with members having different
; kinds of type.

; struct IntFloat {
;   int a;
;   float b;
; };
;
; int SA;
; float SB;
;
; void int_float_struct(struct IntFloat *A) {
;   int SumA;
;   float SumB;
;   for (unsigned i = 0; i < 1024; i++)  {
;     SumA += A[i].a;
;     SumB += A[i].b;
;   }
;   SA = SumA;
;   SB = SumB;
; }

; CHECK-LABEL: @int_float_struct(
; CHECK: %wide.vec = load <8 x i32>, <8 x i32>* %{{.*}}, align 4
; CHECK: %[[V0:.*]] = shufflevector <8 x i32> %wide.vec, <8 x i32> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
; CHECK: %[[V1:.*]] = shufflevector <8 x i32> %wide.vec, <8 x i32> undef, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
; CHECK: bitcast <4 x i32> %[[V1]] to <4 x float>
; CHECK: add <4 x i32>
; CHECK: fadd fast <4 x float>

%struct.IntFloat = type { i32, float }

@SA = common global i32 0, align 4
@SB = common global float 0.000000e+00, align 4

define void @int_float_struct(%struct.IntFloat* nocapture readonly %A) #0 {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  store i32 %add, i32* @SA, align 4
  store float %add3, float* @SB, align 4
  ret void

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %SumB.014 = phi float [ undef, %entry ], [ %add3, %for.body ]
  %SumA.013 = phi i32 [ undef, %entry ], [ %add, %for.body ]
  %a = getelementptr inbounds %struct.IntFloat, %struct.IntFloat* %A, i64 %indvars.iv, i32 0
  %tmp = load i32, i32* %a, align 4
  %add = add nsw i32 %tmp, %SumA.013
  %b = getelementptr inbounds %struct.IntFloat, %struct.IntFloat* %A, i64 %indvars.iv, i32 1
  %tmp1 = load float, float* %b, align 4
  %add3 = fadd fast float %SumB.014, %tmp1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; Check vectorization of interleaved access groups in the presence of
; dependences (PR27626). The following tests check that we don't reorder
; dependent loads and stores when generating code for interleaved access
; groups. Stores should be scalarized because the required code motion would
; break dependences, and the remaining interleaved load groups should have
; gaps.

; PR27626_0: Ensure a strided store is not moved after a dependent (zero
;            distance) strided load.

; void PR27626_0(struct pair *p, int z, int n) {
;   for (int i = 0; i < n; i++) {
;     p[i].x = z;
;     p[i].y = p[i].x;
;   }
; }

; CHECK-LABEL: @PR27626_0(
; CHECK: vector.ph:
; CHECK:   %n.mod.vf = and i64 %[[N:.+]], 3
; CHECK:   %[[IsZero:[a-zA-Z0-9]+]] = icmp eq i64 %n.mod.vf, 0
; CHECK:   %[[R:[a-zA-Z0-9]+]] = select i1 %[[IsZero]], i64 4, i64 %n.mod.vf
; CHECK:   %n.vec = sub nsw i64 %[[N]], %[[R]]
; CHECK: vector.body:
; CHECK:   %[[L1:.+]] = load <8 x i32>, <8 x i32>* {{.*}}
; CHECK:   %[[X1:.+]] = extractelement <8 x i32> %[[L1]], i32 0
; CHECK:   store i32 %[[X1]], {{.*}}
; CHECK:   %[[X2:.+]] = extractelement <8 x i32> %[[L1]], i32 2
; CHECK:   store i32 %[[X2]], {{.*}}
; CHECK:   %[[X3:.+]] = extractelement <8 x i32> %[[L1]], i32 4
; CHECK:   store i32 %[[X3]], {{.*}}
; CHECK:   %[[X4:.+]] = extractelement <8 x i32> %[[L1]], i32 6
; CHECK:   store i32 %[[X4]], {{.*}}

%pair.i32 = type { i32, i32 }
define void @PR27626_0(%pair.i32 *%p, i32 %z, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ %i.next, %for.body ], [ 0, %entry ]
  %p_i.x = getelementptr inbounds %pair.i32, %pair.i32* %p, i64 %i, i32 0
  %p_i.y = getelementptr inbounds %pair.i32, %pair.i32* %p, i64 %i, i32 1
  store i32 %z, i32* %p_i.x, align 4
  %0 = load i32, i32* %p_i.x, align 4
  store i32 %0, i32 *%p_i.y, align 4
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

; PR27626_1: Ensure a strided load is not moved before a dependent (zero
;            distance) strided store.

; void PR27626_1(struct pair *p, int n) {
;   int s = 0;
;   for (int i = 0; i < n; i++) {
;     p[i].y = p[i].x;
;     s += p[i].y
;   }
; }

; CHECK-LABEL: @PR27626_1(
; CHECK: vector.ph:
; CHECK:   %n.mod.vf = and i64 %[[N:.+]], 3
; CHECK:   %[[IsZero:[a-zA-Z0-9]+]] = icmp eq i64 %n.mod.vf, 0
; CHECK:   %[[R:[a-zA-Z0-9]+]] = select i1 %[[IsZero]], i64 4, i64 %n.mod.vf
; CHECK:   %n.vec = sub nsw i64 %[[N]], %[[R]]
; CHECK: vector.body:
; CHECK:   %[[Phi:.+]] = phi <4 x i32> [ zeroinitializer, %vector.ph ], [ {{.*}}, %vector.body ]
; CHECK:   %[[L1:.+]] = load <8 x i32>, <8 x i32>* {{.*}}
; CHECK:   %[[X1:.+]] = extractelement <8 x i32> %[[L1:.+]], i32 0
; CHECK:   store i32 %[[X1:.+]], {{.*}}
; CHECK:   %[[X2:.+]] = extractelement <8 x i32> %[[L1:.+]], i32 2
; CHECK:   store i32 %[[X2:.+]], {{.*}}
; CHECK:   %[[X3:.+]] = extractelement <8 x i32> %[[L1:.+]], i32 4
; CHECK:   store i32 %[[X3:.+]], {{.*}}
; CHECK:   %[[X4:.+]] = extractelement <8 x i32> %[[L1:.+]], i32 6
; CHECK:   store i32 %[[X4:.+]], {{.*}}
; CHECK:   %[[L2:.+]] = load <8 x i32>, <8 x i32>* {{.*}}
; CHECK:   %[[S1:.+]] = shufflevector <8 x i32> %[[L2]], <8 x i32> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
; CHECK:   add <4 x i32> %[[S1]], %[[Phi]]

define i32 @PR27626_1(%pair.i32 *%p, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ %i.next, %for.body ], [ 0, %entry ]
  %s = phi i32 [ %2, %for.body ], [ 0, %entry ]
  %p_i.x = getelementptr inbounds %pair.i32, %pair.i32* %p, i64 %i, i32 0
  %p_i.y = getelementptr inbounds %pair.i32, %pair.i32* %p, i64 %i, i32 1
  %0 = load i32, i32* %p_i.x, align 4
  store i32 %0, i32* %p_i.y, align 4
  %1 = load i32, i32* %p_i.y, align 4
  %2 = add nsw i32 %1, %s
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  %3 = phi i32 [ %2, %for.body ]
  ret i32 %3
}

; PR27626_2: Ensure a strided store is not moved after a dependent (negative
;            distance) strided load.

; void PR27626_2(struct pair *p, int z, int n) {
;   for (int i = 0; i < n; i++) {
;     p[i].x = z;
;     p[i].y = p[i - 1].x;
;   }
; }

; CHECK-LABEL: @PR27626_2(
; CHECK: vector.ph:
; CHECK:   %n.mod.vf = and i64 %[[N:.+]], 3
; CHECK:   %[[IsZero:[a-zA-Z0-9]+]] = icmp eq i64 %n.mod.vf, 0
; CHECK:   %[[R:[a-zA-Z0-9]+]] = select i1 %[[IsZero]], i64 4, i64 %n.mod.vf
; CHECK:   %n.vec = sub nsw i64 %[[N]], %[[R]]
; CHECK: vector.body:
; CHECK:   %[[L1:.+]] = load <8 x i32>, <8 x i32>* {{.*}}
; CHECK:   %[[X1:.+]] = extractelement <8 x i32> %[[L1]], i32 0
; CHECK:   store i32 %[[X1]], {{.*}}
; CHECK:   %[[X2:.+]] = extractelement <8 x i32> %[[L1]], i32 2
; CHECK:   store i32 %[[X2]], {{.*}}
; CHECK:   %[[X3:.+]] = extractelement <8 x i32> %[[L1]], i32 4
; CHECK:   store i32 %[[X3]], {{.*}}
; CHECK:   %[[X4:.+]] = extractelement <8 x i32> %[[L1]], i32 6
; CHECK:   store i32 %[[X4]], {{.*}}

define void @PR27626_2(%pair.i32 *%p, i64 %n, i32 %z) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ %i.next, %for.body ], [ 0, %entry ]
  %i_minus_1 = add nuw nsw i64 %i, -1
  %p_i.x = getelementptr inbounds %pair.i32, %pair.i32* %p, i64 %i, i32 0
  %p_i_minus_1.x = getelementptr inbounds %pair.i32, %pair.i32* %p, i64 %i_minus_1, i32 0
  %p_i.y = getelementptr inbounds %pair.i32, %pair.i32* %p, i64 %i, i32 1
  store i32 %z, i32* %p_i.x, align 4
  %0 = load i32, i32* %p_i_minus_1.x, align 4
  store i32 %0, i32 *%p_i.y, align 4
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

; PR27626_3: Ensure a strided load is not moved before a dependent (negative
;            distance) strided store.

; void PR27626_3(struct pair *p, int z, int n) {
;   for (int i = 0; i < n; i++) {
;     p[i + 1].y = p[i].x;
;     s += p[i].y;
;   }
; }

; CHECK-LABEL: @PR27626_3(
; CHECK: vector.ph:
; CHECK:   %n.mod.vf = and i64 %[[N:.+]], 3
; CHECK:   %[[IsZero:[a-zA-Z0-9]+]] = icmp eq i64 %n.mod.vf, 0
; CHECK:   %[[R:[a-zA-Z0-9]+]] = select i1 %[[IsZero]], i64 4, i64 %n.mod.vf
; CHECK:   %n.vec = sub nsw i64 %[[N]], %[[R]]
; CHECK: vector.body:
; CHECK:   %[[Phi:.+]] = phi <4 x i32> [ zeroinitializer, %vector.ph ], [ {{.*}}, %vector.body ]
; CHECK:   %[[L1:.+]] = load <8 x i32>, <8 x i32>* {{.*}}
; CHECK:   %[[X1:.+]] = extractelement <8 x i32> %[[L1:.+]], i32 0
; CHECK:   store i32 %[[X1:.+]], {{.*}}
; CHECK:   %[[X2:.+]] = extractelement <8 x i32> %[[L1:.+]], i32 2
; CHECK:   store i32 %[[X2:.+]], {{.*}}
; CHECK:   %[[X3:.+]] = extractelement <8 x i32> %[[L1:.+]], i32 4
; CHECK:   store i32 %[[X3:.+]], {{.*}}
; CHECK:   %[[X4:.+]] = extractelement <8 x i32> %[[L1:.+]], i32 6
; CHECK:   store i32 %[[X4:.+]], {{.*}}
; CHECK:   %[[L2:.+]] = load <8 x i32>, <8 x i32>* {{.*}}
; CHECK:   %[[S1:.+]] = shufflevector <8 x i32> %[[L2]], <8 x i32> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
; CHECK:   add <4 x i32> %[[S1]], %[[Phi]]

define i32 @PR27626_3(%pair.i32 *%p, i64 %n, i32 %z) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ %i.next, %for.body ], [ 0, %entry ]
  %s = phi i32 [ %2, %for.body ], [ 0, %entry ]
  %i_plus_1 = add nuw nsw i64 %i, 1
  %p_i.x = getelementptr inbounds %pair.i32, %pair.i32* %p, i64 %i, i32 0
  %p_i.y = getelementptr inbounds %pair.i32, %pair.i32* %p, i64 %i, i32 1
  %p_i_plus_1.y = getelementptr inbounds %pair.i32, %pair.i32* %p, i64 %i_plus_1, i32 1
  %0 = load i32, i32* %p_i.x, align 4
  store i32 %0, i32* %p_i_plus_1.y, align 4
  %1 = load i32, i32* %p_i.y, align 4
  %2 = add nsw i32 %1, %s
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  %3 = phi i32 [ %2, %for.body ]
  ret i32 %3
}

; PR27626_4: Ensure we form an interleaved group for strided stores in the
;            presence of a write-after-write dependence. We create a group for
;            (2) and (3) while excluding (1).

; void PR27626_4(int *a, int x, int y, int z, int n) {
;   for (int i = 0; i < n; i += 2) {
;     a[i] = x;      // (1)
;     a[i] = y;      // (2)
;     a[i + 1] = z;  // (3)
;   }
; }

; CHECK-LABEL: @PR27626_4(
; CHECK: vector.ph:
; CHECK:   %[[INS_Y:.+]] = insertelement <4 x i32> undef, i32 %y, i32 0
; CHECK:   %[[SPLAT_Y:.+]] = shufflevector <4 x i32> %[[INS_Y]], <4 x i32> undef, <4 x i32> zeroinitializer
; CHECK:   %[[INS_Z:.+]] = insertelement <4 x i32> undef, i32 %z, i32 0
; CHECK:   %[[SPLAT_Z:.+]] = shufflevector <4 x i32> %[[INS_Z]], <4 x i32> undef, <4 x i32> zeroinitializer
; CHECK: vector.body:
; CHECK:   store i32 %x, {{.*}}
; CHECK:   store i32 %x, {{.*}}
; CHECK:   store i32 %x, {{.*}}
; CHECK:   store i32 %x, {{.*}}
; CHECK:   %[[VEC:.+]] = shufflevector <4 x i32> %[[SPLAT_Y]], <4 x i32> %[[SPLAT_Z]], <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
; CHECK:   store <8 x i32> %[[VEC]], {{.*}}

define void @PR27626_4(i32 *%a, i32 %x, i32 %y, i32 %z, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ %i.next, %for.body ], [ 0, %entry ]
  %i_plus_1 = add i64 %i, 1
  %a_i = getelementptr inbounds i32, i32* %a, i64 %i
  %a_i_plus_1 = getelementptr inbounds i32, i32* %a, i64 %i_plus_1
  store i32 %x, i32* %a_i, align 4
  store i32 %y, i32* %a_i, align 4
  store i32 %z, i32* %a_i_plus_1, align 4
  %i.next = add nuw nsw i64 %i, 2
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

; PR27626_5: Ensure we do not form an interleaved group for strided stores in
;            the presence of a write-after-write dependence.

; void PR27626_5(int *a, int x, int y, int z, int n) {
;   for (int i = 3; i < n; i += 2) {
;     a[i - 1] = x;
;     a[i - 3] = y;
;     a[i] = z;
;   }
; }

; CHECK-LABEL: @PR27626_5(
; CHECK: vector.body:
; CHECK:   store i32 %x, {{.*}}
; CHECK:   store i32 %x, {{.*}}
; CHECK:   store i32 %x, {{.*}}
; CHECK:   store i32 %x, {{.*}}
; CHECK:   store i32 %y, {{.*}}
; CHECK:   store i32 %y, {{.*}}
; CHECK:   store i32 %y, {{.*}}
; CHECK:   store i32 %y, {{.*}}
; CHECK:   store i32 %z, {{.*}}
; CHECK:   store i32 %z, {{.*}}
; CHECK:   store i32 %z, {{.*}}
; CHECK:   store i32 %z, {{.*}}

define void @PR27626_5(i32 *%a, i32 %x, i32 %y, i32 %z, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ %i.next, %for.body ], [ 3, %entry ]
  %i_minus_1 = sub i64 %i, 1
  %i_minus_3 = sub i64 %i_minus_1, 2
  %a_i = getelementptr inbounds i32, i32* %a, i64 %i
  %a_i_minus_1 = getelementptr inbounds i32, i32* %a, i64 %i_minus_1
  %a_i_minus_3 = getelementptr inbounds i32, i32* %a, i64 %i_minus_3
  store i32 %x, i32* %a_i_minus_1, align 4
  store i32 %y, i32* %a_i_minus_3, align 4
  store i32 %z, i32* %a_i, align 4
  %i.next = add nuw nsw i64 %i, 2
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

; PR34743: Ensure that a cast which needs to sink after a load that belongs to
; an interleaved group, indeeded gets sunk.

; void PR34743(short *a, int *b, int n) {
;   for (int i = 0, iv = 0; iv < n; i++, iv += 2) {
;     b[i] = a[iv] * a[iv+1] * a[iv+2];
;   }
; }

; CHECK-LABEL: @PR34743(
; CHECK: vector.body:
; CHECK:   %vector.recur = phi <4 x i16> [ %vector.recur.init, %vector.ph ], [ %[[VSHUF1:.+]], %vector.body ]
; CHECK:   %wide.vec = load <8 x i16>
; CHECK:   %[[VSHUF0:.+]] = shufflevector <8 x i16> %wide.vec, <8 x i16> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
; CHECK:   %[[VSHUF1:.+]] = shufflevector <8 x i16> %wide.vec, <8 x i16> undef, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
; CHECK:   %[[VSHUF:.+]] = shufflevector <4 x i16> %vector.recur, <4 x i16> %[[VSHUF1]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; CHECK:   sext <4 x i16> %[[VSHUF0]] to <4 x i32>
; CHECK:   sext <4 x i16> %[[VSHUF]] to <4 x i32>
; CHECK:   sext <4 x i16> %[[VSHUF1]] to <4 x i32>
; CHECK:   mul nsw <4 x i32>
; CHECK:   mul nsw <4 x i32>

define void @PR34743(i16* %a, i32* %b, i64 %n) {
entry:
  %.pre = load i16, i16* %a
  br label %loop

loop:
  %0 = phi i16 [ %.pre, %entry ], [ %load2, %loop ]
  %iv = phi i64 [ 0, %entry ], [ %iv2, %loop ]
  %i = phi i64 [ 0, %entry ], [ %i1, %loop ]
  %conv = sext i16 %0 to i32
  %i1 = add nuw nsw i64 %i, 1
  %iv1 = add nuw nsw i64 %iv, 1
  %iv2 = add nuw nsw i64 %iv, 2
  %gep1 = getelementptr inbounds i16, i16* %a, i64 %iv1
  %load1 = load i16, i16* %gep1, align 4
  %conv1 = sext i16 %load1 to i32
  %gep2 = getelementptr inbounds i16, i16* %a, i64 %iv2
  %load2 = load i16, i16* %gep2, align 4
  %conv2 = sext i16 %load2 to i32
  %mul01 = mul nsw i32 %conv, %conv1
  %mul012 = mul nsw i32 %mul01, %conv2
  %arrayidx5 = getelementptr inbounds i32, i32* %b, i64 %i
  store i32 %mul012, i32* %arrayidx5
  %exitcond = icmp eq i64 %iv, %n
  br i1 %exitcond, label %end, label %loop

end:
  ret void
}

attributes #0 = { "unsafe-fp-math"="true" }
