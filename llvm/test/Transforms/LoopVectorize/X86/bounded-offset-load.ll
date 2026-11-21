; RUN: opt -S -passes=loop-vectorize -force-vector-width=4 -force-vector-interleave=1 -mtriple=x86_64 -mcpu=haswell %s | FileCheck %s
; REQUIRES: x86-registered-target

@table = external constant [65 x i64], align 8
declare void @llvm.assume(i1)

; CHECK-LABEL: define void @bounded_offset(
; CHECK: vector.body:
; CHECK: load <4 x i64>, ptr {{.*}}, align 8
; CHECK-NOT: @llvm.masked.load
; CHECK: load <4 x i64>, ptr {{.*}}, align 8
; CHECK-NOT: @llvm.masked.load
; CHECK: middle.block:
define void @bounded_offset(ptr noalias %out, ptr noalias %conditions, i64 %index) {
entry:
  %in_range = icmp ult i64 %index, 58
  call void @llvm.assume(i1 %in_range)
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %latch ]
  %cond_ptr = getelementptr inbounds i64, ptr %conditions, i64 %i
  %cond = load i64, ptr %cond_ptr, align 8
  %take = icmp ne i64 %cond, 0
  br i1 %take, label %load_table, label %latch

load_table:
  %table_index = add nuw i64 %index, %i
  %table_ptr = getelementptr inbounds [65 x i64], ptr @table, i64 0, i64 %table_index
  %value = load i64, ptr %table_ptr, align 8
  br label %latch

latch:
  %result = phi i64 [ 0, %loop ], [ %value, %load_table ]
  %out_ptr = getelementptr inbounds i64, ptr %out, i64 %i
  store i64 %result, ptr %out_ptr, align 8
  %next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %next, 8
  br i1 %done, label %exit, label %loop

exit:
  ret void
}


; CHECK-LABEL: define void @signed_bounded_offset(
; CHECK: vector.body:
; CHECK: load <4 x i64>, ptr {{.*}}, align 8
; CHECK-NOT: @llvm.masked.load
; CHECK: load <4 x i64>, ptr {{.*}}, align 8
; CHECK-NOT: @llvm.masked.load
; CHECK: middle.block:
define void @signed_bounded_offset(ptr noalias %out, ptr noalias %conditions, i64 %index) {
entry:
  %nonnegative = icmp sge i64 %index, 0
  call void @llvm.assume(i1 %nonnegative)
  %in_range = icmp slt i64 %index, 58
  call void @llvm.assume(i1 %in_range)
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %latch ]
  %cond_ptr = getelementptr inbounds i64, ptr %conditions, i64 %i
  %cond = load i64, ptr %cond_ptr, align 8
  %take = icmp ne i64 %cond, 0
  br i1 %take, label %load_table, label %latch

load_table:
  %table_index = add nuw i64 %index, %i
  %table_ptr = getelementptr inbounds [65 x i64], ptr @table, i64 0, i64 %table_index
  %value = load i64, ptr %table_ptr, align 8
  br label %latch

latch:
  %result = phi i64 [ 0, %loop ], [ %value, %load_table ]
  %out_ptr = getelementptr inbounds i64, ptr %out, i64 %i
  store i64 %result, ptr %out_ptr, align 8
  %next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %next, 8
  br i1 %done, label %exit, label %loop

exit:
  ret void
}


; CHECK-LABEL: define void @unbounded_offset(
; CHECK: vector.body:
; CHECK: call <4 x i64> @llvm.masked.load
; CHECK: middle.block:
define void @unbounded_offset(ptr noalias %out, ptr noalias %conditions, i64 %index) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %latch ]
  %cond_ptr = getelementptr inbounds i64, ptr %conditions, i64 %i
  %cond = load i64, ptr %cond_ptr, align 8
  %take = icmp ne i64 %cond, 0
  br i1 %take, label %load_table, label %latch

load_table:
  %table_index = add nuw i64 %index, %i
  %table_ptr = getelementptr inbounds [65 x i64], ptr @table, i64 0, i64 %table_index
  %value = load i64, ptr %table_ptr, align 8
  br label %latch

latch:
  %result = phi i64 [ 0, %loop ], [ %value, %load_table ]
  %out_ptr = getelementptr inbounds i64, ptr %out, i64 %i
  store i64 %result, ptr %out_ptr, align 8
  %next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %next, 8
  br i1 %done, label %exit, label %loop

exit:
  ret void
}


; CHECK-LABEL: define void @one_past_offset(
; CHECK: vector.body:
; CHECK: call <4 x i64> @llvm.masked.load
; CHECK: middle.block:
define void @one_past_offset(ptr noalias %out, ptr noalias %conditions, i64 %index) {
entry:
  %in_range = icmp ult i64 %index, 59
  call void @llvm.assume(i1 %in_range)
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %latch ]
  %cond_ptr = getelementptr inbounds i64, ptr %conditions, i64 %i
  %cond = load i64, ptr %cond_ptr, align 8
  %take = icmp ne i64 %cond, 0
  br i1 %take, label %load_table, label %latch

load_table:
  %table_index = add nuw i64 %index, %i
  %table_ptr = getelementptr inbounds [65 x i64], ptr @table, i64 0, i64 %table_index
  %value = load i64, ptr %table_ptr, align 8
  br label %latch

latch:
  %result = phi i64 [ 0, %loop ], [ %value, %load_table ]
  %out_ptr = getelementptr inbounds i64, ptr %out, i64 %i
  store i64 %result, ptr %out_ptr, align 8
  %next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %next, 8
  br i1 %done, label %exit, label %loop

exit:
  ret void
}


; CHECK-LABEL: define void @branch_local_bound(
; CHECK: vector.body:
; CHECK: call <4 x i64> @llvm.masked.load
; CHECK: middle.block:
define void @branch_local_bound(ptr noalias %out, ptr noalias %conditions, i64 %index) {
entry:
  %in_range = icmp ult i64 %index, 58
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %latch ]
  %cond_ptr = getelementptr inbounds i64, ptr %conditions, i64 %i
  %cond = load i64, ptr %cond_ptr, align 8
  %take = icmp ne i64 %cond, 0
  br i1 %take, label %load_table, label %latch

load_table:
  call void @llvm.assume(i1 %in_range)
  %table_index = add nuw i64 %index, %i
  %table_ptr = getelementptr inbounds [65 x i64], ptr @table, i64 0, i64 %table_index
  %value = load i64, ptr %table_ptr, align 8
  br label %latch

latch:
  %result = phi i64 [ 0, %loop ], [ %value, %load_table ]
  %out_ptr = getelementptr inbounds i64, ptr %out, i64 %i
  store i64 %result, ptr %out_ptr, align 8
  %next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %next, 8
  br i1 %done, label %exit, label %loop

exit:
  ret void
}


; CHECK-LABEL: define void @negative_offset(
; CHECK: vector.body:
; CHECK: call <4 x i64> @llvm.masked.load
; CHECK: middle.block:
define void @negative_offset(ptr noalias %out, ptr noalias %conditions, i64 %index) {
entry:
  %in_range = icmp ugt i64 %index, -65
  call void @llvm.assume(i1 %in_range)
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %latch ]
  %cond_ptr = getelementptr inbounds i64, ptr %conditions, i64 %i
  %cond = load i64, ptr %cond_ptr, align 8
  %take = icmp ne i64 %cond, 0
  br i1 %take, label %load_table, label %latch

load_table:
  %table_index = add nuw i64 %index, %i
  %table_ptr = getelementptr inbounds [65 x i64], ptr @table, i64 0, i64 %table_index
  %value = load i64, ptr %table_ptr, align 8
  br label %latch

latch:
  %result = phi i64 [ 0, %loop ], [ %value, %load_table ]
  %out_ptr = getelementptr inbounds i64, ptr %out, i64 %i
  store i64 %result, ptr %out_ptr, align 8
  %next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %next, 8
  br i1 %done, label %exit, label %loop

exit:
  ret void
}


; CHECK-LABEL: define void @misaligned_offset(
; CHECK: vector.body:
; CHECK: call <4 x i64> @llvm.masked.load
; CHECK: middle.block:
define void @misaligned_offset(ptr noalias %out, ptr noalias %conditions, i64 %index) {
entry:
  %in_range = icmp ult i64 %index, 58
  call void @llvm.assume(i1 %in_range)
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %latch ]
  %cond_ptr = getelementptr inbounds i64, ptr %conditions, i64 %i
  %cond = load i64, ptr %cond_ptr, align 8
  %take = icmp ne i64 %cond, 0
  br i1 %take, label %load_table, label %latch

load_table:
  %step = shl nuw i64 %i, 3
  %table_index = add nuw i64 %index, %step
  %table_ptr = getelementptr inbounds i8, ptr @table, i64 %table_index
  %value = load i64, ptr %table_ptr, align 8
  br label %latch

latch:
  %result = phi i64 [ 0, %loop ], [ %value, %load_table ]
  %out_ptr = getelementptr inbounds i64, ptr %out, i64 %i
  store i64 %result, ptr %out_ptr, align 8
  %next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %next, 8
  br i1 %done, label %exit, label %loop

exit:
  ret void
}


; CHECK-LABEL: define void @unknown_allocation(
; CHECK: vector.body:
; CHECK: call <4 x i64> @llvm.masked.load
; CHECK: middle.block:
define void @unknown_allocation(ptr noalias %out, ptr noalias %conditions, i64 %index, ptr %unknown) {
entry:
  %in_range = icmp ult i64 %index, 58
  call void @llvm.assume(i1 %in_range)
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %latch ]
  %cond_ptr = getelementptr inbounds i64, ptr %conditions, i64 %i
  %cond = load i64, ptr %cond_ptr, align 8
  %take = icmp ne i64 %cond, 0
  br i1 %take, label %load_table, label %latch

load_table:
  %table_index = add nuw i64 %index, %i
  %table_ptr = getelementptr inbounds [65 x i64], ptr %unknown, i64 0, i64 %table_index
  %value = load i64, ptr %table_ptr, align 8
  br label %latch

latch:
  %result = phi i64 [ 0, %loop ], [ %value, %load_table ]
  %out_ptr = getelementptr inbounds i64, ptr %out, i64 %i
  store i64 %result, ptr %out_ptr, align 8
  %next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %next, 8
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; The start address is a pointer umin plus an integer, so its last operand is
; the integer and there is no single base object.
; CHECK-LABEL: define void @pointer_min_start(
; CHECK: vector.body:
; CHECK: pred.load.if:
; CHECK: middle.block:
define void @pointer_min_start(ptr noalias %out, ptr noalias %conditions, i32 noundef %a, i32 noundef %b, i64 noundef %n) {
entry:
  %a.ext = zext i32 %a to i64
  %b.ext = zext i32 %b to i64
  %p = inttoptr i64 %a.ext to ptr
  %q = inttoptr i64 %b.ext to ptr
  %p.first = icmp ult ptr %p, %q
  %min = select i1 %p.first, ptr %p, ptr %q
  %start = getelementptr i8, ptr %min, i64 %n
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %latch ]
  %cond_ptr = getelementptr inbounds i8, ptr %conditions, i64 %i
  %cond = load i8, ptr %cond_ptr, align 1
  %take = icmp ne i8 %cond, 0
  br i1 %take, label %load_start, label %latch

load_start:
  %ptr = getelementptr i8, ptr %start, i64 %i
  %value = load i8, ptr %ptr, align 1
  br label %latch

latch:
  %result = phi i8 [ 0, %loop ], [ %value, %load_start ]
  %out_ptr = getelementptr inbounds i8, ptr %out, i64 %i
  store i8 %result, ptr %out_ptr, align 1
  %next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %next, 8
  br i1 %done, label %exit, label %loop

exit:
  ret void
}
