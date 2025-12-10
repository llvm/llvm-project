; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

define void @range_prefetch_reg(ptr %src, i64 %metadata) {
  ; CHECK: write argument to llvm.aarch64.range.prefetch.reg must be 0 or 1
  ; CHECK-NEXT: call void @llvm.aarch64.range.prefetch.reg(ptr %src, i32 2, i32 0, i64 %metadata)
  call void @llvm.aarch64.range.prefetch.reg(ptr %src, i32 2, i32 0, i64 %metadata)

  ; CHECK-NEXT: stream argument to llvm.aarch64.range.prefetch.reg must be 0 or 1
  ; CHECK-NEXT: call void @llvm.aarch64.range.prefetch.reg(ptr %src, i32 0, i32 2, i64 %metadata)
  call void @llvm.aarch64.range.prefetch.reg(ptr %src, i32 0, i32 2, i64 %metadata)

  ret void
}

define void @range_prefetch_imm(ptr %src) {
  ; CHECK: write argument to llvm.aarch64.range.prefetch.imm must be 0 or 1
  ; CHECK-NEXT: call void @llvm.aarch64.range.prefetch.imm(ptr %src, i32 2, i32 0, i32 0, i32 1, i32 0, i64 0)
  call void @llvm.aarch64.range.prefetch.imm(ptr %src, i32 2, i32 0, i32 0, i32 1, i32 0, i64 0)

  ; CHECK-NEXT: stream argument to llvm.aarch64.range.prefetch.imm must be 0 or 1
  ; CHECK-NEXT: call void @llvm.aarch64.range.prefetch.imm(ptr %src, i32 0, i32 2, i32 0, i32 1, i32 0, i64 0)
  call void @llvm.aarch64.range.prefetch.imm(ptr %src, i32 0, i32 2, i32 0, i32 1, i32 0, i64 0)

  ; CHECK-NEXT: length argument to llvm.aarch64.range.prefetch.imm must be -2048 - 2040
  ; CHECK-NEXT: call void @llvm.aarch64.range.prefetch.imm(ptr %src, i32 0, i32 0, i32 -2049, i32 1, i32 0, i64 0)
  call void @llvm.aarch64.range.prefetch.imm(ptr %src, i32 0, i32 0, i32 -2049, i32 1, i32 0, i64 0)

  ; CHECK-NEXT: length argument to llvm.aarch64.range.prefetch.imm must be -2048 - 2040
  ; CHECK-NEXT: call void @llvm.aarch64.range.prefetch.imm(ptr %src, i32 0, i32 0, i32 2041, i32 1, i32 0, i64 0)
  call void @llvm.aarch64.range.prefetch.imm(ptr %src, i32 0, i32 0, i32 2041, i32 1, i32 0, i64 0)

  ; CHECK-NEXT: count argument to llvm.aarch64.range.prefetch.imm must be < 65537
  ; CHECK-NEXT: call void @llvm.aarch64.range.prefetch.imm(ptr %src, i32 0, i32 0, i32 0, i32 0, i32 0, i64 0)
  call void @llvm.aarch64.range.prefetch.imm(ptr %src, i32 0, i32 0, i32 0, i32 0, i32 0, i64 0)

  ; CHECK-NEXT: count argument to llvm.aarch64.range.prefetch.imm must be < 65537
  ; CHECK-NEXT: call void @llvm.aarch64.range.prefetch.imm(ptr %src, i32 0, i32 0, i32 0, i32 65537, i32 0, i64 0)
  call void @llvm.aarch64.range.prefetch.imm(ptr %src, i32 0, i32 0, i32 0, i32 65537, i32 0, i64 0)

  ; CHECK-NEXT: stride argument to llvm.aarch64.range.prefetch.imm must be -2048 - 2040
  ; CHECK-NEXT: call void @llvm.aarch64.range.prefetch.imm(ptr %src, i32 0, i32 0, i32 0, i32 1, i32 -2049, i64 0)
  call void @llvm.aarch64.range.prefetch.imm(ptr %src, i32 0, i32 0, i32 0, i32 1, i32 -2049, i64 0)

  ; CHECK-NEXT: stride argument to llvm.aarch64.range.prefetch.imm must be -2048 - 2040
  ; CHECK-NEXT: call void @llvm.aarch64.range.prefetch.imm(ptr %src, i32 0, i32 0, i32 0, i32 1, i32 2041, i64 0)
  call void @llvm.aarch64.range.prefetch.imm(ptr %src, i32 0, i32 0, i32 0, i32 1, i32 2041, i64 0)

  ret void
}
