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

; CHECK-LABEL: @test_struct_load4(
; CHECK: %wide.vec = load <16 x i32>, <16 x i32>* {{.*}}, align 4
; CHECK: shufflevector <16 x i32> %wide.vec, <16 x i32> undef, <4 x i32> <i32 0, i32 4, i32 8, i32 12>
; CHECK: shufflevector <16 x i32> %wide.vec, <16 x i32> undef, <4 x i32> <i32 1, i32 5, i32 9, i32 13>
; CHECK: shufflevector <16 x i32> %wide.vec, <16 x i32> undef, <4 x i32> <i32 2, i32 6, i32 10, i32 14>
; CHECK: shufflevector <16 x i32> %wide.vec, <16 x i32> undef, <4 x i32> <i32 3, i32 7, i32 11, i32 15>
; CHECK: add nsw <4 x i32>
; CHECK: sub <4 x i32>
; CHECK: add nsw <4 x i32>
; CHECK: sub <4 x i32>

%struct.ST4 = type { i32, i32, i32, i32 }

define i32 @test_struct_load4(%struct.ST4* nocapture readonly %S) {
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
; CHECK: %wide.vec = load <8 x i32>, <8 x i32>* {{.*}}, align 4
; CHECK: shufflevector <8 x i32> %wide.vec, <8 x i32> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
; CHECK: shufflevector <4 x i32> {{.*}}, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
; CHECK: shufflevector <8 x i32> %wide.vec, <8 x i32> undef, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
; CHECK: shufflevector <4 x i32> {{.*}}, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
; CHECK: add nsw <4 x i32>
; CHECK: sub nsw <4 x i32>
; CHECK: shufflevector <4 x i32> {{.*}}, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
; CHECK: shufflevector <4 x i32> {{.*}}, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
; CHECK: %interleaved.vec = shufflevector <4 x i32> {{.*}}, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
; CHECK: store <8 x i32> %interleaved.vec, <8 x i32>* %{{.*}}, align 4

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
; (missing the load of odd elements).

; void even_load(int *A, int *B) {
;  for (unsigned i = 0; i < 1024; i+=2)
;     B[i/2] = A[i] * 2;
; }

; CHECK-LABEL: @even_load(
; CHECK-NOT: %wide.vec = load <8 x i32>, <8 x i32>* %{{.*}}, align 4
; CHECK-NOT: %strided.vec = shufflevector <8 x i32> %wide.vec, <8 x i32> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>

define void @even_load(i32* noalias nocapture readonly %A, i32* noalias nocapture %B) {
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
; CHECK: add nsw <4 x i32>
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

attributes #0 = { "unsafe-fp-math"="true" }
