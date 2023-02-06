; RUN: opt -passes=loop-vectorize -force-vector-width=4 -force-vector-interleave=2 -S %s | FileCheck %s
; RUN: opt -passes=loop-vectorize -force-vector-width=1 -force-vector-interleave=2 -S %s | FileCheck %s

; Test cases for selecting the index with the minimum value.

define i64 @test_vectorize_select_umin_idx(ptr %src) {
; CHECK-LABEL: @test_vectorize_select_umin_idx(
; CHECK-NOT:   vector.body:
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %min.idx = phi i64 [ 0, %entry ], [ %min.idx.next, %loop ]
  %min.val = phi i64 [ 0, %entry ], [ %min.val.next, %loop ]
  %gep = getelementptr i64, ptr %src, i64 %iv
  %l = load i64, ptr %gep
  %cmp = icmp ugt i64 %min.val, %l
  %min.val.next = tail call i64 @llvm.umin.i64(i64 %min.val, i64 %l)
  %min.idx.next = select i1 %cmp, i64 %iv, i64 %min.idx
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 0
  br i1 %exitcond.not, label %exit, label %loop

exit:
  %res = phi i64 [ %min.idx.next, %loop ]
  ret i64 %res
}

define i64 @test_vectorize_select_umin_idx_min_ops_switched(ptr %src) {
; CHECK-LABEL: @test_vectorize_select_umin_idx_min_ops_switched(
; CHECK-NOT:   vector.body:
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %min.idx = phi i64 [ 0, %entry ], [ %min.idx.next, %loop ]
  %min.val = phi i64 [ 0, %entry ], [ %min.val.next, %loop ]
  %gep = getelementptr i64, ptr %src, i64 %iv
  %l = load i64, ptr %gep
  %cmp = icmp ugt i64 %min.val, %l
  %min.val.next = tail call i64 @llvm.umin.i64(i64 %l, i64 %min.val)
  %min.idx.next = select i1 %cmp, i64 %iv, i64 %min.idx
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 0
  br i1 %exitcond.not, label %exit, label %loop

exit:
  %res = phi i64 [ %min.idx.next, %loop ]
  ret i64 %res
}

define i64 @test_not_vectorize_select_no_min_reduction(ptr %src) {
; CHECK-LABEL: @test_not_vectorize_select_no_min_reduction(
; CHECK-NOT:   vector.body:
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %min.idx = phi i64 [ 0, %entry ], [ %min.idx.next, %loop ]
  %min.val = phi i64 [ 0, %entry ], [ %min.val.next, %loop ]
  %gep = getelementptr i64, ptr %src, i64 %iv
  %l = load i64, ptr %gep
  %cmp = icmp ugt i64 %min.val, %l
  %min.val.next = add i64 %l, 1
  %foo = call i64 @llvm.umin.i64(i64 %min.val, i64 %l)
  %min.idx.next = select i1 %cmp, i64 %iv, i64 %min.idx
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 0
  br i1 %exitcond.not, label %exit, label %loop

exit:
  %res = phi i64 [ %min.idx.next, %loop ]
  ret i64 %res
}


define i64 @test_not_vectorize_cmp_value(i64 %x) {
; CHECK-LABEL: @test_not_vectorize_cmp_value(
; CHECK-NOT:   vector.body:
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %min.idx = phi i64 [ 0, %entry ], [ %min.idx.next, %loop ]
  %min.val = phi i64 [ 0, %entry ], [ %min.val.next, %loop ]
  %cmp = icmp ugt i64 %min.val, %x
  %min.val.next = tail call i64 @llvm.umin.i64(i64 %min.val, i64 0)
  %min.idx.next = select i1 %cmp, i64 %iv, i64 %min.idx
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 0
  br i1 %exitcond.not, label %exit, label %loop

exit:
  %res = phi i64 [ %min.idx.next, %loop ]
  ret i64 %res
}

define i32 @test_vectorize_select_umin_idx_with_trunc() {
; CHECK-LABEL: @test_vectorize_select_umin_idx_with_trunc(
; CHECK-NOT:   vector.body:
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %min.idx = phi i32 [ 0, %entry ], [ %min.idx.next, %loop ]
  %min.val = phi i64 [ 0, %entry ], [ %min.val.next, %loop ]
  %cmp = icmp ugt i64 %min.val, 0
  %min.val.next = tail call i64 @llvm.umin.i64(i64 %min.val, i64 0)
  %trunc = trunc i64 %iv to i32
  %min.idx.next = select i1 %cmp, i32 %trunc, i32 %min.idx
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 0
  br i1 %exitcond.not, label %exit, label %loop

exit:
  %res = phi i32 [ %min.idx.next, %loop ]
  ret i32 %res
}

define ptr @test_with_ptr_index(ptr %start, ptr %end) {
; CHECK-LABEL: @test_with_ptr_index(
; CHECK-NOT:   vector.body:
;
entry:
  br label %loop

loop:
  %iv = phi ptr [ %start, %entry ], [ %iv.next, %loop ]
  %min.idx = phi ptr [ null, %entry ], [ %min.idx.next, %loop ]
  %min.val = phi i64 [ 0, %entry ], [ %min.val.next, %loop ]
  %cmp7.us = icmp ult i64 0, 0
  %min.val.next = tail call i64 @llvm.umin.i64(i64 %min.val, i64 0)
  %min.idx.next = select i1 %cmp7.us, ptr %iv, ptr %min.idx
  %iv.next = getelementptr i32, ptr %iv, i64 1
  %exitcond.not = icmp eq ptr %iv.next, %end
  br i1 %exitcond.not, label %exit, label %loop

exit:
  %res = phi ptr [ %min.idx.next, %loop ]
  ret ptr %res
}

define void @pointer_index(ptr %start) {
; CHECK-LABEL: @pointer_index(
; CHECK-NOT:   vector.body:
;
entry:
  br label %loop

loop:
  %ptr.iv = phi ptr [ %start, %entry ], [ %ptr.iv.next, %loop ]
  %ptr.idx = phi ptr [ %start, %entry ], [ %ptr.select, %loop ]
  %cmp.i.i.i.i2531 = icmp ult i16 0, 0
  %ptr.select = select i1 %cmp.i.i.i.i2531, ptr %ptr.iv, ptr %ptr.idx
  %ptr.iv.next = getelementptr inbounds i16, ptr %ptr.iv, i64 1
  %cmp.i.i10.not.i.i.i = icmp eq ptr %ptr.iv.next, null
  br i1 %cmp.i.i10.not.i.i.i, label %exit, label %loop

exit:
  ret void
}

define ptr @pointer_index_2(ptr %start, ptr %end) {
; CHECK-LABEL: @pointer_index_2(
; CHECK-NOT:   vector.body:
;
entry:
  br label %loop

loop:
  %min.val  = phi i16 [ 0, %entry ], [ %min.val.next, %loop ]
  %ptr.iv = phi ptr [ %start, %entry ], [ %ptr.iv.next, %loop ]
  %min.idx = phi ptr [ %start, %entry ], [ %min.idx.next, %loop ]
  %cmp.i.i.i.i = icmp ult i16 0, %min.val
  %min.val.next = call i16 @llvm.umin.i16(i16 0, i16 %min.val)
  %min.idx.next = select i1 %cmp.i.i.i.i, ptr %ptr.iv, ptr %min.idx
  %ptr.iv.next = getelementptr inbounds i16, ptr %ptr.iv, i64 1
  %exit.cond = icmp eq ptr %ptr.iv.next, %end
  br i1 %exit.cond, label %exit, label %loop

exit:
  %res = phi ptr [ %min.idx.next, %loop ]
  ret ptr %res
}

define i64 @test_no_vectorize_select_iv_decrement(ptr %src) {
; CHECK-LABEL: @test_no_vectorize_select_iv_decrement(
; CHECK-NOT:   vector.body:
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 1000, %entry ], [ %iv.next, %loop ]
  %min.idx = phi i64 [ 0, %entry ], [ %min.idx.next, %loop ]
  %min.val = phi i64 [ 0, %entry ], [ %min.val.next, %loop ]
  %gep = getelementptr i64, ptr %src, i64 %iv
  %l = load i64, ptr %gep
  %cmp = icmp ugt i64 %min.val, %l
  %min.val.next = tail call i64 @llvm.umin.i64(i64 %min.val, i64 %l)
  %min.idx.next = select i1 %cmp, i64 %iv, i64 %min.idx
  %iv.next = add nuw nsw i64 %iv, -1
  %exitcond.not = icmp eq i64 %iv.next, 0
  br i1 %exitcond.not, label %exit, label %loop

exit:
  %res = phi i64 [ %min.idx.next, %loop ]
  ret i64 %res
}

define i64 @test_no_vectorize_select_iv_sub(ptr %src) {
; CHECK-LABEL: @test_no_vectorize_select_iv_sub(
; CHECK-NOT:   vector.body:
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 1000, %entry ], [ %iv.next, %loop ]
  %min.idx = phi i64 [ 0, %entry ], [ %min.idx.next, %loop ]
  %min.val = phi i64 [ 0, %entry ], [ %min.val.next, %loop ]
  %gep = getelementptr i64, ptr %src, i64 %iv
  %l = load i64, ptr %gep
  %cmp = icmp ugt i64 %min.val, %l
  %min.val.next = tail call i64 @llvm.umin.i64(i64 %min.val, i64 %l)
  %min.idx.next = select i1 %cmp, i64 %iv, i64 %min.idx
  %iv.next = sub i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 0
  br i1 %exitcond.not, label %exit, label %loop

exit:
  %res = phi i64 [ %min.idx.next, %loop ]
  ret i64 %res
}

define i64 @test_no_vectorize_select_iv_mul(ptr %src) {
; CHECK-LABEL: @test_no_vectorize_select_iv_mul(
; CHECK-NOT:   vector.body:
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 1, %entry ], [ %iv.next, %loop ]
  %min.idx = phi i64 [ 0, %entry ], [ %min.idx.next, %loop ]
  %min.val = phi i64 [ 0, %entry ], [ %min.val.next, %loop ]
  %gep = getelementptr i64, ptr %src, i64 %iv
  %l = load i64, ptr %gep
  %cmp = icmp ugt i64 %min.val, %l
  %min.val.next = tail call i64 @llvm.umin.i64(i64 %min.val, i64 %l)
  %min.idx.next = select i1 %cmp, i64 %iv, i64 %min.idx
  %iv.next = mul i64 %iv, 2
  %exitcond.not = icmp eq i64 %iv.next, 128
  br i1 %exitcond.not, label %exit, label %loop

exit:
  %res = phi i64 [ %min.idx.next, %loop ]
  ret i64 %res
}

declare i64 @llvm.umin.i64(i64, i64)
declare i16 @llvm.umin.i16(i16, i16)
