; RUN: opt -passes=loop-idiom -S -loop-idiom-enable-memset-cont-check < %s | FileCheck --check-prefixes=CHECK-MSPOS,CHECK-MSNEG %s
; RUN: opt -passes=loop-idiom -S -loop-idiom-enable-memcpy-cont-check < %s | FileCheck --check-prefixes=CHECK-MCNEG,CHECK-MEMMOVE %s
; RUN: opt -passes=loop-idiom -S -verify-each -loop-idiom-enable-memset-cont-check < %s -o /dev/null
; RUN: opt -passes=loop-idiom -S -verify-each -loop-idiom-enable-memcpy-cont-check < %s -o /dev/null
; RUN: opt -passes=loop-idiom -S \
; RUN:   -loop-idiom-enable-memset-cont-check \
; RUN:   -loop-idiom-force-memset-pattern-intrinsic < %s \
; RUN:   | FileCheck --check-prefixes=CHECK-PATTERN %s
; RUN: opt -passes=loop-idiom -S -verify-each \
; RUN:   -loop-idiom-enable-memset-cont-check \
; RUN:   -loop-idiom-force-memset-pattern-intrinsic < %s -o /dev/null

target datalayout = "e-p:32:32"

; Positive memset, i16 index type.
; CHECK-MSPOS-LABEL: @memset_i16(
; CHECK-MSPOS: loop.idiom.cont.cond:
; CHECK-MSPOS: call void @llvm.memset.p0.i32(ptr align 4 %scevgep, i8 0, i32 16, i1 false)
; CHECK-MSPOS: call void @llvm.memset.p0.i32(ptr align 4 %scevgep, i8 0, i32 %1, i1 false)

define void @memset_i16(ptr %arr, i16 %ubnd) {
entry:
  %cmp0 = icmp slt i16 %ubnd, 1
  br i1 %cmp0, label %exit, label %outer.preheader

outer.preheader:
  %base = getelementptr i8, ptr %arr, i32 -4
  br label %outer

outer:
  %oy = phi i16 [ 1, %outer.preheader ], [ %oy.next, %outer.latch ]
  %row.off = shl nsw i16 %oy, 2
  br label %inner

inner:
  %iy = phi i16 [ 1, %outer ], [ %iy.next, %inner ]
  %sum = add nuw nsw i16 %iy, %row.off
  %p = getelementptr i32, ptr %base, i16 %sum
  store i32 0, ptr %p, align 4
  %iy.next = add nuw nsw i16 %iy, 1
  %inner.done = icmp eq i16 %iy, %ubnd
  br i1 %inner.done, label %outer.latch, label %inner

outer.latch:
  %oy.next = add nuw nsw i16 %oy, 1
  %outer.done = icmp eq i16 %oy, 100
  br i1 %outer.done, label %exit, label %outer

exit:
  ret void
}

; Negative memset cases: memset formation expected, no contiguity diamond.
; CHECK-MSNEG-LABEL: @memset_neg_stride_parent_step(
; CHECK-MSNEG-NOT: loop.idiom.cont.cond:
; CHECK-MSNEG: call void @llvm.memset

define void @memset_neg_stride_parent_step(ptr %arr, i16 %ubnd) {
entry:
  %cmp0 = icmp slt i16 %ubnd, 1
  br i1 %cmp0, label %exit, label %outer.preheader

outer.preheader:
  %base = getelementptr i8, ptr %arr, i32 -4
  br label %outer

outer:
  %oy = phi i16 [ 100, %outer.preheader ], [ %oy.next, %outer.latch ]
  %row.off = shl nsw i16 %oy, 2
  br label %inner

inner:
  %iy = phi i16 [ 1, %outer ], [ %iy.next, %inner ]
  %sum = add nuw nsw i16 %iy, %row.off
  %p = getelementptr i32, ptr %base, i16 %sum
  store i32 0, ptr %p, align 4
  %iy.next = add nuw nsw i16 %iy, 1
  %inner.done = icmp eq i16 %iy, %ubnd
  br i1 %inner.done, label %outer.latch, label %inner

outer.latch:
  %oy.next = add nsw i16 %oy, -1
  %outer.done = icmp eq i16 %oy, 1
  br i1 %outer.done, label %exit, label %outer

exit:
  ret void
}

; CHECK-MSNEG-LABEL: @memset_step_not_multiple_store_size(
; CHECK-MSNEG-NOT: loop.idiom.cont.cond:
; CHECK-MSNEG: call void @llvm.memset

define void @memset_step_not_multiple_store_size(ptr %arr, i16 %ubnd) {
entry:
  %cmp0 = icmp slt i16 %ubnd, 1
  br i1 %cmp0, label %exit, label %outer.preheader

outer.preheader:
  %base = getelementptr i8, ptr %arr, i32 -8
  br label %outer

outer:
  %oy = phi i16 [ 1, %outer.preheader ], [ %oy.next, %outer.latch ]
  %row.off = mul nsw i16 %oy, 12
  br label %inner

inner:
  %iy = phi i16 [ 1, %outer ], [ %iy.next, %inner ]
  %elem.off = mul nuw nsw i16 %iy, 8
  %sum = add nuw nsw i16 %elem.off, %row.off
  %pi8 = getelementptr i8, ptr %base, i16 %sum
  store i64 0, ptr %pi8, align 1
  %iy.next = add nuw nsw i16 %iy, 1
  %inner.done = icmp eq i16 %iy, %ubnd
  br i1 %inner.done, label %outer.latch, label %inner

outer.latch:
  %oy.next = add nuw nsw i16 %oy, 1
  %outer.done = icmp eq i16 %oy, 50
  br i1 %outer.done, label %exit, label %outer

exit:
  ret void
}

; CHECK-MSNEG-LABEL: @memset_constant_inner_extent(
; CHECK-MSNEG-NOT: loop.idiom.cont.cond:
; CHECK-MSNEG: call void @llvm.memset

define void @memset_constant_inner_extent(ptr %arr) {
entry:
  br label %outer.preheader

outer.preheader:
  %base = getelementptr i8, ptr %arr, i32 -4
  br label %outer

outer:
  %oy = phi i16 [ 1, %outer.preheader ], [ %oy.next, %outer.latch ]
  %row.off = shl nsw i16 %oy, 2
  br label %inner

inner:
  %iy = phi i16 [ 1, %outer ], [ %iy.next, %inner ]
  %sum = add nuw nsw i16 %iy, %row.off
  %p = getelementptr i32, ptr %base, i16 %sum
  store i32 0, ptr %p, align 4
  %iy.next = add nuw nsw i16 %iy, 1
  %inner.done = icmp eq i16 %iy, 4
  br i1 %inner.done, label %outer.latch, label %inner

outer.latch:
  %oy.next = add nuw nsw i16 %oy, 1
  %outer.done = icmp eq i16 %oy, 100
  br i1 %outer.done, label %exit, label %outer

exit:
  ret void
}

; Negative memcpy cases: memcpy formation expected, no contiguity diamond.
; CHECK-MCNEG-LABEL: @memcpy_neg_stride_parent_step(
; CHECK-MCNEG-NOT: loop.idiom.cont.cond:
; CHECK-MCNEG: call void @llvm.memcpy

define void @memcpy_neg_stride_parent_step(ptr noalias %dst, ptr noalias %src, i16 %ubnd) {
entry:
  %cmp0 = icmp slt i16 %ubnd, 1
  br i1 %cmp0, label %exit, label %outer.preheader

outer.preheader:
  %dst.base = getelementptr i8, ptr %dst, i32 -4
  %src.base = getelementptr i8, ptr %src, i32 -4
  br label %outer

outer:
  %oy = phi i16 [ 100, %outer.preheader ], [ %oy.next, %outer.latch ]
  %row.off = shl nsw i16 %oy, 2
  br label %inner

inner:
  %iy = phi i16 [ 1, %outer ], [ %iy.next, %inner ]
  %sum = add nuw nsw i16 %iy, %row.off
  %dp = getelementptr i32, ptr %dst.base, i16 %sum
  %sp = getelementptr i32, ptr %src.base, i16 %sum
  %v = load i32, ptr %sp, align 1
  store i32 %v, ptr %dp, align 4
  %iy.next = add nuw nsw i16 %iy, 1
  %inner.done = icmp eq i16 %iy, %ubnd
  br i1 %inner.done, label %outer.latch, label %inner

outer.latch:
  %oy.next = add nsw i16 %oy, -1
  %outer.done = icmp eq i16 %oy, 1
  br i1 %outer.done, label %exit, label %outer

exit:
  ret void
}

; CHECK-MCNEG-LABEL: @memcpy_step_not_multiple_store_size(
; CHECK-MCNEG-NOT: loop.idiom.cont.cond:
; CHECK-MCNEG: call void @llvm.memcpy

define void @memcpy_step_not_multiple_store_size(ptr noalias %dst, ptr noalias %src, i16 %ubnd) {
entry:
  %cmp0 = icmp slt i16 %ubnd, 1
  br i1 %cmp0, label %exit, label %outer.preheader

outer.preheader:
  %dst.base = getelementptr i8, ptr %dst, i32 -8
  %src.base = getelementptr i8, ptr %src, i32 -8
  br label %outer

outer:
  %oy = phi i16 [ 1, %outer.preheader ], [ %oy.next, %outer.latch ]
  %row.off = mul nsw i16 %oy, 12
  br label %inner

inner:
  %iy = phi i16 [ 1, %outer ], [ %iy.next, %inner ]
  %elem.off = mul nuw nsw i16 %iy, 8
  %sum = add nuw nsw i16 %elem.off, %row.off
  %dp = getelementptr i8, ptr %dst.base, i16 %sum
  %sp = getelementptr i8, ptr %src.base, i16 %sum
  %v = load i64, ptr %sp, align 1
  store i64 %v, ptr %dp, align 1
  %iy.next = add nuw nsw i16 %iy, 1
  %inner.done = icmp eq i16 %iy, %ubnd
  br i1 %inner.done, label %outer.latch, label %inner

outer.latch:
  %oy.next = add nuw nsw i16 %oy, 1
  %outer.done = icmp eq i16 %oy, 50
  br i1 %outer.done, label %exit, label %outer

exit:
  ret void
}

; CHECK-MCNEG-LABEL: @memcpy_constant_inner_extent(
; CHECK-MCNEG-NOT: loop.idiom.cont.cond:
; CHECK-MCNEG: call void @llvm.memcpy

define void @memcpy_constant_inner_extent(ptr noalias %dst, ptr noalias %src) {
entry:
  br label %outer.preheader

outer.preheader:
  %dst.base = getelementptr i8, ptr %dst, i32 -4
  %src.base = getelementptr i8, ptr %src, i32 -4
  br label %outer

outer:
  %oy = phi i16 [ 1, %outer.preheader ], [ %oy.next, %outer.latch ]
  %row.off = shl nsw i16 %oy, 2
  br label %inner

inner:
  %iy = phi i16 [ 1, %outer ], [ %iy.next, %inner ]
  %sum = add nuw nsw i16 %iy, %row.off
  %dp = getelementptr i32, ptr %dst.base, i16 %sum
  %sp = getelementptr i32, ptr %src.base, i16 %sum
  %v = load i32, ptr %sp, align 1
  store i32 %v, ptr %dp, align 4
  %iy.next = add nuw nsw i16 %iy, 1
  %inner.done = icmp eq i16 %iy, 4
  br i1 %inner.done, label %outer.latch, label %inner

outer.latch:
  %oy.next = add nuw nsw i16 %oy, 1
  %outer.done = icmp eq i16 %oy, 100
  br i1 %outer.done, label %exit, label %outer

exit:
  ret void
}

; Exclusions: memset.pattern and memmove paths skip the contiguity diamond.
; CHECK-PATTERN-LABEL: @memset_pattern_no_diamond(
; CHECK-PATTERN-NOT: loop.idiom.cont.cond:
; CHECK-PATTERN: call void @llvm.experimental.memset.pattern
; CHECK-PATTERN-NOT: loop.idiom.cont.cond:

; CHECK-MEMMOVE-LABEL: @memmove_no_diamond(
; CHECK-MEMMOVE-NOT: loop.idiom.cont.cond:
; CHECK-MEMMOVE: call void @llvm.memmove
; CHECK-MEMMOVE-NOT: loop.idiom.cont.cond:

define void @memset_pattern_no_diamond(ptr %arr, i16 %ubnd) {
entry:
  %cmp0 = icmp slt i16 %ubnd, 1
  br i1 %cmp0, label %exit, label %outer.preheader

outer.preheader:
  %base = getelementptr i8, ptr %arr, i32 -4
  br label %outer

outer:
  %oy = phi i16 [ 1, %outer.preheader ], [ %oy.next, %outer.latch ]
  %row.off = shl nsw i16 %oy, 2
  br label %inner

inner:
  %iy = phi i16 [ 1, %outer ], [ %iy.next, %inner ]
  %sum = add nuw nsw i16 %iy, %row.off
  %p = getelementptr i32, ptr %base, i16 %sum
  store i32 305419896, ptr %p, align 4
  %iy.next = add nuw nsw i16 %iy, 1
  %inner.done = icmp eq i16 %iy, %ubnd
  br i1 %inner.done, label %outer.latch, label %inner

outer.latch:
  %oy.next = add nuw nsw i16 %oy, 1
  %outer.done = icmp eq i16 %oy, 100
  br i1 %outer.done, label %exit, label %outer

exit:
  ret void
}

define void @memmove_no_diamond(ptr %Src, i64 %Size) {
bb.nph:
  br label %for.body

for.body:
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %for.body ]
  %Step = add nuw nsw i64 %indvar, 1
  %SrcI = getelementptr i8, ptr %Src, i64 %Step
  %DestI = getelementptr i8, ptr %Src, i64 %indvar
  %V = load i8, ptr %SrcI, align 1
  store i8 %V, ptr %DestI, align 1
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, %Size
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}
