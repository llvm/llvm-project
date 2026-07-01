; RUN: opt -passes=loop-idiom -S -loop-idiom-enable-memset-cont-check < %s | FileCheck --check-prefixes=CHECK-MS %s
; RUN: opt -passes=loop-idiom -S -loop-idiom-enable-memcpy-cont-check < %s | FileCheck --check-prefixes=CHECK-MC,CHECK-MC-POS %s
; RUN: opt -passes=loop-idiom -S -verify-each -loop-idiom-enable-memset-cont-check < %s -o /dev/null
; RUN: opt -passes=loop-idiom -S -verify-each -loop-idiom-enable-memcpy-cont-check < %s -o /dev/null

; Address space 1 (non-integral index): positive memcpy diamond and lossy
; stride guards for memset/memcpy.
;
; CHECK-MC-POS-LABEL: @memcpy_non_integral_as(
; CHECK-MC-POS: loop.idiom.cont.cond:
; CHECK-MC-POS: br i1 {{.*}}, label %loop.idiom.cont.then, label %loop.idiom.cont.else, !unpredictable
; CHECK-MC-POS: call void @llvm.memcpy.p1.p1.i32({{.*}} i32 32, i1 false)
; CHECK-MC-POS: call void @llvm.memcpy.p1.p1.i32({{.*}} i32 %ubnd, i1 false)

; CHECK-MS-LABEL: @memset_lossy(
; CHECK-MS-NOT: loop.idiom.cont.cond:
; CHECK-MS: call void @llvm.memset.p1.i32(
; CHECK-MS-SAME: i32 %ubnd

; CHECK-MC-LABEL: @memcpy_lossy(
; CHECK-MC-NOT: loop.idiom.cont.cond:
; CHECK-MC: call void @llvm.memcpy.p1.p1.i32(
; CHECK-MC-SAME: i32 %ubnd

target datalayout = "e-p:64:64-p1:64:64:64:32-n8:16:32:64"

define void @memcpy_non_integral_as(ptr addrspace(1) noalias %dst, ptr addrspace(1) noalias %src, i32 %ubnd) {
entry:
  %cmp0 = icmp slt i32 %ubnd, 1
  br i1 %cmp0, label %exit, label %outer.preheader

outer.preheader:
  %dst.base = getelementptr i8, ptr addrspace(1) %dst, i64 -1
  %src.base = getelementptr i8, ptr addrspace(1) %src, i64 -1
  br label %outer

outer:
  %oy = phi i64 [ 1, %outer.preheader ], [ %oy.next, %outer.latch ]
  %row.off = mul nuw nsw i64 %oy, 32
  br label %inner

inner:
  %iy = phi i32 [ 1, %outer ], [ %iy.next, %inner ]
  %iy64 = zext i32 %iy to i64
  %sum = add nuw nsw i64 %iy64, %row.off
  %src.p = getelementptr i8, ptr addrspace(1) %src.base, i64 %sum
  %dst.p = getelementptr i8, ptr addrspace(1) %dst.base, i64 %sum
  %v = load i8, ptr addrspace(1) %src.p, align 1
  store i8 %v, ptr addrspace(1) %dst.p, align 1
  %iy.next = add nuw nsw i32 %iy, 1
  %inner.done = icmp eq i32 %iy, %ubnd
  br i1 %inner.done, label %outer.latch, label %inner

outer.latch:
  %oy.next = add nuw nsw i64 %oy, 1
  %outer.done = icmp eq i64 %oy, 8
  br i1 %outer.done, label %exit, label %outer

exit:
  ret void
}

define void @memset_lossy(ptr addrspace(1) %arr, i32 %ubnd) {
entry:
  %cmp0 = icmp slt i32 %ubnd, 1
  br i1 %cmp0, label %exit, label %outer.preheader

outer.preheader:
  %base = getelementptr i8, ptr addrspace(1) %arr, i64 -1
  br label %outer

outer:
  %oy = phi i64 [ 1, %outer.preheader ], [ %oy.next, %outer.latch ]
  %row.off = mul nuw nsw i64 %oy, 1099511627776
  br label %inner

inner:
  %iy = phi i32 [ 1, %outer ], [ %iy.next, %inner ]
  %iy64 = zext i32 %iy to i64
  %sum = add nuw nsw i64 %iy64, %row.off
  %p = getelementptr i8, ptr addrspace(1) %base, i64 %sum
  store i8 0, ptr addrspace(1) %p, align 1
  %iy.next = add nuw nsw i32 %iy, 1
  %inner.done = icmp eq i32 %iy, %ubnd
  br i1 %inner.done, label %outer.latch, label %inner

outer.latch:
  %oy.next = add nuw nsw i64 %oy, 1
  %outer.done = icmp eq i64 %oy, 8
  br i1 %outer.done, label %exit, label %outer

exit:
  ret void
}

define void @memcpy_lossy(ptr addrspace(1) noalias %dst, ptr addrspace(1) noalias %src, i32 %ubnd) {
entry:
  %cmp0 = icmp slt i32 %ubnd, 1
  br i1 %cmp0, label %exit, label %outer.preheader

outer.preheader:
  %dst.base = getelementptr i8, ptr addrspace(1) %dst, i64 -1
  %src.base = getelementptr i8, ptr addrspace(1) %src, i64 -1
  br label %outer

outer:
  %oy = phi i64 [ 1, %outer.preheader ], [ %oy.next, %outer.latch ]
  %row.off = mul nuw nsw i64 %oy, 1099511627776
  br label %inner

inner:
  %iy = phi i32 [ 1, %outer ], [ %iy.next, %inner ]
  %iy64 = zext i32 %iy to i64
  %sum = add nuw nsw i64 %iy64, %row.off
  %src.p = getelementptr i8, ptr addrspace(1) %src.base, i64 %sum
  %dst.p = getelementptr i8, ptr addrspace(1) %dst.base, i64 %sum
  %v = load i8, ptr addrspace(1) %src.p, align 1
  store i8 %v, ptr addrspace(1) %dst.p, align 1
  %iy.next = add nuw nsw i32 %iy, 1
  %inner.done = icmp eq i32 %iy, %ubnd
  br i1 %inner.done, label %outer.latch, label %inner

outer.latch:
  %oy.next = add nuw nsw i64 %oy, 1
  %outer.done = icmp eq i64 %oy, 8
  br i1 %outer.done, label %exit, label %outer

exit:
  ret void
}
