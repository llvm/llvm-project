; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

define void @range_prefetch(ptr %src, i64 %metadata) {
  ; CHECK: write argument to llvm.aarch64.range.prefetch must be 0 or 1
  ; CHECK-NEXT: call void @llvm.aarch64.range.prefetch(ptr %src, i32 2, i32 0, i64 %metadata)
  call void @llvm.aarch64.range.prefetch(ptr %src, i32 2, i32 0, i64 %metadata)

  ; CHECK-NEXT: stream argument to llvm.aarch64.range.prefetch must be 0 or 1
  ; CHECK-NEXT: call void @llvm.aarch64.range.prefetch(ptr %src, i32 0, i32 2, i64 %metadata)
  call void @llvm.aarch64.range.prefetch(ptr %src, i32 0, i32 2, i64 %metadata)

  ret void
}

declare void @llvm.aarch64.stshh.atomic.store.p0(ptr, i64, i32 immarg, i32 immarg, i32 immarg)

define void @stshh_atomic_store_order_non_imm(ptr %p, i64 %v, i32 %arg0) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg0
  ; CHECK-NEXT: call void @llvm.aarch64.stshh.atomic.store.p0(ptr %p, i64 %v, i32 %arg0, i32 0, i32 64)
  call void @llvm.aarch64.stshh.atomic.store.p0(ptr %p, i64 %v, i32 %arg0, i32 0, i32 64)
  ret void
}

define void @stshh_atomic_store_policy_non_imm(ptr %p, i64 %v, i32 %arg0) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg0
  ; CHECK-NEXT: call void @llvm.aarch64.stshh.atomic.store.p0(ptr %p, i64 %v, i32 0, i32 %arg0, i32 64)
  call void @llvm.aarch64.stshh.atomic.store.p0(ptr %p, i64 %v, i32 0, i32 %arg0, i32 64)
  ret void
}

define void @stshh_atomic_store_size_non_imm(ptr %p, i64 %v, i32 %arg0) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg0
  ; CHECK-NEXT: call void @llvm.aarch64.stshh.atomic.store.p0(ptr %p, i64 %v, i32 0, i32 0, i32 %arg0)
  call void @llvm.aarch64.stshh.atomic.store.p0(ptr %p, i64 %v, i32 0, i32 0, i32 %arg0)
  ret void
}

define void @stshh_atomic_store_order_out_of_range(ptr %p, i64 %v) {
  ; CHECK: order argument to llvm.aarch64.stshh.atomic.store must be 0, 3 or 5
  ; CHECK-NEXT: call void @llvm.aarch64.stshh.atomic.store.p0(ptr %p, i64 %v, i32 1, i32 0, i32 64)
  call void @llvm.aarch64.stshh.atomic.store.p0(ptr %p, i64 %v, i32 1, i32 0, i32 64)
  ret void
}

define void @stshh_atomic_store_policy_out_of_range(ptr %p, i64 %v) {
  ; CHECK: policy argument to llvm.aarch64.stshh.atomic.store must be 0 or 1
  ; CHECK-NEXT: call void @llvm.aarch64.stshh.atomic.store.p0(ptr %p, i64 %v, i32 0, i32 2, i32 64)
  call void @llvm.aarch64.stshh.atomic.store.p0(ptr %p, i64 %v, i32 0, i32 2, i32 64)
  ret void
}

define void @stshh_atomic_store_size_out_of_range(ptr %p, i64 %v) {
  ; CHECK: size argument to llvm.aarch64.stshh.atomic.store must be 8, 16, 32 or 64
  ; CHECK-NEXT: call void @llvm.aarch64.stshh.atomic.store.p0(ptr %p, i64 %v, i32 0, i32 0, i32 0)
  call void @llvm.aarch64.stshh.atomic.store.p0(ptr %p, i64 %v, i32 0, i32 0, i32 0)
  ret void
}
