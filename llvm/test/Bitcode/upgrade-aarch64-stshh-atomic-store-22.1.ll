; RUN: llvm-dis < %S/upgrade-aarch64-stshh-atomic-store-22.1.bc | FileCheck %s

; CHECK-LABEL: define void @relaxed_i8(
; CHECK: %[[TRUNC:.+]] = trunc i64 %v to i8
; CHECK-NEXT: store atomic i8 %[[TRUNC]], ptr %p monotonic, align 1
define void @relaxed_i8(ptr %p, i64 %v) {
  call void @llvm.aarch64.stshh.atomic.store.p0(ptr %p, i64 %v, i32 0, i32 1, i32 8)
  ret void
}

; CHECK-LABEL: define void @release_i32(
; CHECK: %[[TRUNC:.+]] = trunc i64 %v to i32
; CHECK-NEXT: store atomic i32 %[[TRUNC]], ptr %p release, align 4
define void @release_i32(ptr %p, i64 %v) {
  call void @llvm.aarch64.stshh.atomic.store.p0(ptr %p, i64 %v, i32 3, i32 0, i32 32)
  ret void
}

; CHECK-LABEL: define void @seq_cst_i64(
; CHECK: store atomic i64 %v, ptr %p seq_cst, align 8
define void @seq_cst_i64(ptr %p, i64 %v) {
  call void @llvm.aarch64.stshh.atomic.store.p0(ptr %p, i64 %v, i32 5, i32 1, i32 64)
  ret void
}

declare void @llvm.aarch64.stshh.atomic.store.p0(ptr, i64, i32 immarg, i32 immarg, i32 immarg)
