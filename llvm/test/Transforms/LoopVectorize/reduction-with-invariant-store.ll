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
; CHECK-NOT: vector.body
define void @reduc_store(i32* %dst, i32* readonly %src) {
entry:
  %gep.dst = getelementptr inbounds i32, i32* %dst, i64 42
  store i32 0, i32* %gep.dst, align 4
  br label %for.body

for.body:
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %gep.src = getelementptr inbounds i32, i32* %src, i64 %iv
  %0 = load i32, i32* %gep.src, align 4
  %add = add nsw i32 %sum, %0
  store i32 %add, i32* %gep.dst, align 4
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
; CHECK-NOT: vector.body
define void @reduc_store_fadd_fast(float* %dst, float* readonly %src) {
entry:
  %gep.dst = getelementptr inbounds float, float* %dst, i64 42
  store float 0.000000e+00, float* %gep.dst, align 4
  br label %for.body

for.body:
  %sum = phi float [ 0.000000e+00, %entry ], [ %add, %for.body ]
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %gep.src = getelementptr inbounds float, float* %src, i64 %iv
  %0 = load float, float* %gep.src, align 4
  %add = fadd fast float %sum, %0
  store float %add, float* %gep.dst, align 4
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
define void @reduc_store_load(i32* %dst, i32* readonly %src, i32* noalias %dst.2) {
entry:
  %gep.dst = getelementptr inbounds i32, i32* %dst, i64 42
  store i32 0, i32* %gep.dst, align 4
  br label %for.body

for.body:
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %gep.src = getelementptr inbounds i32, i32* %src, i64 %iv
  %0 = load i32, i32* %gep.src, align 4
  %add = add nsw i32 %sum, %0
  %lv = load i32, i32* %gep.dst
  %gep.dst.2  = getelementptr inbounds i32, i32* %dst.2, i64 %iv
  store i32 %lv, i32* %gep.dst.2, align 4
  store i32 %add, i32* %gep.dst, align 4
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
define void @reduc_cond_store(i32* %t, i32* readonly %x, i32* readonly %y) {
entry:
  store i32 0, i32* %t, align 4
  br label %for.body

for.body:
  %sum = phi i32 [ 0, %entry ], [ %sum.2, %if.end ]
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %if.end ]
  %gep.y = getelementptr inbounds i32, i32* %y, i64 %iv
  %0 = load i32, i32* %gep.y, align 4
  %gep.x = getelementptr inbounds i32, i32* %x, i64 %iv
  %1 = load i32, i32* %gep.x, align 4
  %diff = sub nsw i32 %0, %1
  %cmp2 = icmp sgt i32 %diff, 0
  br i1 %cmp2, label %if.then, label %if.end

if.then:
  %sum.1 = add nsw i32 %diff, %sum
  store i32 %sum.1, i32* %t, align 4
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
; CHECK-NOT: vector.body
define void @reduc_store_inside_unrolled(i32* %dst, i32* readonly %src) {
entry:
  %gep.dst = getelementptr inbounds i32, i32* %dst, i64 42
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %sum.1, %for.body ]
  %gep.src = getelementptr inbounds i32, i32* %src, i64 %iv
  %0 = load i32, i32* %gep.src, align 4
  %sum.1 = add nsw i32 %0, %sum
  store i32 %sum.1, i32* %gep.dst, align 4
  %1 = or i64 %iv, 1
  %gep.src.1 = getelementptr inbounds i32, i32* %src, i64 %1
  %2 = load i32, i32* %gep.src.1, align 4
  %sum.2 = add nsw i32 %2, %sum.1
  store i32 %sum.2, i32* %gep.dst, align 4
  %iv.next = add nuw nsw i64 %iv, 2
  %cmp = icmp slt i64 %iv.next, 1000
  br i1 %cmp, label %for.body, label %exit

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
define void @reduc_double_invariant_store(i32* %dst, i32* %other_dst, i32* readonly %src) {
entry:
  %gep.dst = getelementptr inbounds i32, i32* %dst, i64 42
  %gep.other_dst = getelementptr inbounds i32, i32* %other_dst, i64 42
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %sum.1, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %src, i64 %iv
  %0 = load i32, i32* %arrayidx, align 4
  %sum.1 = add nsw i32 %0, %sum
  store i32 %sum.1, i32* %gep.dst, align 4
  %1 = or i64 %iv, 1
  %arrayidx4 = getelementptr inbounds i32, i32* %src, i64 %1
  %2 = load i32, i32* %arrayidx4, align 4
  %sum.2 = add nsw i32 %2, %sum.1
  store i32 %sum.2, i32* %gep.other_dst, align 4
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
; CHECK-NOT: vector.body
define void @reduc_store_middle_store_predicated(i32* %dst, i32* readonly %src) {
entry:
  %gep.dst = getelementptr inbounds i32, i32* %dst, i64 42
  br label %for.body

for.body:                                         ; preds = %latch, %entry
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %latch ]
  %sum = phi i32 [ 0, %entry ], [ %sum.2, %latch ]
  %gep.src = getelementptr inbounds i32, i32* %src, i64 %iv
  %0 = load i32, i32* %gep.src, align 4
  %sum.1 = add nsw i32 %0, %sum
  %cmp = icmp sgt i32 %0, 0
  br i1 %cmp, label %predicated, label %latch

predicated:                                       ; preds = %for.body
  store i32 %sum.1, i32* %gep.dst, align 4
  br label %latch

latch:                                            ; preds = %predicated, %for.body
  %1 = or i64 %iv, 1
  %gep.src.1 = getelementptr inbounds i32, i32* %src, i64 %1
  %2 = load i32, i32* %gep.src.1, align 4
  %sum.2 = add nsw i32 %2, %sum.1
  store i32 %sum.2, i32* %gep.dst, align 4
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
define void @reduc_store_final_store_predicated(i32* %dst, i32* readonly %src) {
entry:
  %gep.dst = getelementptr inbounds i32, i32* %dst, i64 42
  br label %for.body

for.body:                                         ; preds = %latch, %entry
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %latch ]
  %sum = phi i32 [ 0, %entry ], [ %sum.1, %latch ]
  %arrayidx = getelementptr inbounds i32, i32* %src, i64 %iv
  %0 = load i32, i32* %arrayidx, align 4
  %sum.1 = add nsw i32 %0, %sum
  store i32 %sum.1, i32* %gep.dst, align 4
  %1 = or i64 %iv, 1
  %gep.src.1 = getelementptr inbounds i32, i32* %src, i64 %1
  %2 = load i32, i32* %gep.src.1, align 4
  %sum.2 = add nsw i32 %2, %sum.1
  %cmp1 = icmp sgt i32 %2, 0
  br i1 %cmp1, label %predicated, label %latch

predicated:                                       ; preds = %for.body
  store i32 %sum.2, i32* %gep.dst, align 4
  br label %latch

latch:                                            ; preds = %predicated, %for.body
  %iv.next = add nuw nsw i64 %iv, 2
  %cmp = icmp slt i64 %iv.next, 1000
  br i1 %cmp, label %for.body, label %exit

exit:                                 ; preds = %latch
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
; CHECK-NOT: vector.body
define void @reduc_store_inoutside(i32* %dst, i32* readonly %src) {
entry:
  %gep.dst = getelementptr inbounds i32, i32* %dst, i64 42
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %sum.1, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %src, i64 %iv
  %0 = load i32, i32* %arrayidx, align 4
  %sum.1 = add nsw i32 %0, %sum
  store i32 %sum.1, i32* %gep.dst, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %exit, label %for.body

exit:
  %sum.lcssa = phi i32 [ %sum.1, %for.body ]
  %gep.dst.1 = getelementptr inbounds i32, i32* %dst, i64 43
  store i32 %sum.lcssa, i32* %gep.dst.1, align 4
  ret void
}
