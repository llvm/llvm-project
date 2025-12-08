; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

define void @range_prefetch(ptr %src) {
  ; CHECK: write argument to llvm.aarch64.range.prefetch must be 0 or 1
  ; CHECK-NEXT: call void @llvm.aarch64.range.prefetch(ptr %src, i32 2, i32 0, i32 0, i32 0, i32 1, i32 0)
  call void @llvm.aarch64.range.prefetch(ptr %src, i32 2, i32 0, i32 0, i32 0, i32 1, i32 0)

  ; CHECK-NEXT: stream argument to llvm.aarch64.range.prefetch must be 0 or 1
  ; CHECK-NEXT: call void @llvm.aarch64.range.prefetch(ptr %src, i32 0, i32 2, i32 0, i32 0, i32 1, i32 0)
  call void @llvm.aarch64.range.prefetch(ptr %src, i32 0, i32 2, i32 0, i32 0, i32 1, i32 0)

  ; CHECK-NEXT: reuse distance argument to llvm.aarch64.range.prefetch must be < 16
  ; CHECK-NEXT: call void @llvm.aarch64.range.prefetch(ptr %src, i32 0, i32 0, i32 16, i32 0, i32 1, i32 0)
  call void @llvm.aarch64.range.prefetch(ptr %src, i32 0, i32 0, i32 16, i32 0, i32 1, i32 0)

  ; CHECK-NEXT: stride argument to llvm.aarch64.range.prefetch must be -2048 - 2040
  ; CHECK-NEXT: call void @llvm.aarch64.range.prefetch(ptr %src, i32 0, i32 0, i32 0, i32 -2049, i32 1, i32 0)
  call void @llvm.aarch64.range.prefetch(ptr %src, i32 0, i32 0, i32 0, i32 -2049, i32 1, i32 0)

  ; CHECK-NEXT: stride argument to llvm.aarch64.range.prefetch must be -2048 - 2040
  ; CHECK-NEXT: call void @llvm.aarch64.range.prefetch(ptr %src, i32 0, i32 0, i32 0, i32 2041, i32 1, i32 0)
  call void @llvm.aarch64.range.prefetch(ptr %src, i32 0, i32 0, i32 0, i32 2041, i32 1, i32 0)

  ; CHECK-NEXT: count argument to llvm.aarch64.range.prefetch must be < 65537
  ; CHECK-NEXT: call void @llvm.aarch64.range.prefetch(ptr %src, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.aarch64.range.prefetch(ptr %src, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)

  ; CHECK-NEXT: count argument to llvm.aarch64.range.prefetch must be < 65537
  ; CHECK-NEXT: call void @llvm.aarch64.range.prefetch(ptr %src, i32 0, i32 0, i32 0, i32 0, i32 65537, i32 0)
  call void @llvm.aarch64.range.prefetch(ptr %src, i32 0, i32 0, i32 0, i32 0, i32 65537, i32 0)

  ; CHECK-NEXT: length argument to llvm.aarch64.range.prefetch must be -2048 - 2040
  ; CHECK-NEXT: call void @llvm.aarch64.range.prefetch(ptr %src, i32 0, i32 0, i32 0, i32 0, i32 1, i32 -2049)
  call void @llvm.aarch64.range.prefetch(ptr %src, i32 0, i32 0, i32 0, i32 0, i32 1, i32 -2049)

  ; CHECK-NEXT: length argument to llvm.aarch64.range.prefetch must be -2048 - 2040
  ; CHECK-NEXT: call void @llvm.aarch64.range.prefetch(ptr %src, i32 0, i32 0, i32 0, i32 0, i32 1, i32 2041)
  call void @llvm.aarch64.range.prefetch(ptr %src, i32 0, i32 0, i32 0, i32 0, i32 1, i32 2041)

  ret void
}
