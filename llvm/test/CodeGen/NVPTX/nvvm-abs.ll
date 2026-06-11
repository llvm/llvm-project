; RUN: llc -march=nvptx64 < %s | FileCheck %s

declare i32 @llvm.nvvm.abs.i(i32)
declare i64 @llvm.nvvm.abs.ll(i64)

define i32 @test_nvvm_abs_i(i32 %x) {
; CHECK-LABEL: test_nvvm_abs_i(
; CHECK-NOT: max.s32
; CHECK: abs.s32
; CHECK-NOT: max.s32
; CHECK: ret;
  %r = call i32 @llvm.nvvm.abs.i(i32 %x)
  ret i32 %r
}

define i64 @test_nvvm_abs_ll(i64 %x) {
; CHECK-LABEL: test_nvvm_abs_ll(
; CHECK-NOT: max.s64
; CHECK: abs.s64
; CHECK-NOT: max.s64
; CHECK: ret;
  %r = call i64 @llvm.nvvm.abs.ll(i64 %x)
  ret i64 %r
}
