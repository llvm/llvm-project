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
