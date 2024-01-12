; RUN: llc < %s -march=nvptx -mcpu=sm_50 | FileCheck %s

; CHECK-LABEL: test_mulhi_i16
; CHECK: mul.hi.s16
define i16 @test_mulhi_i16(i16 %x, i16 %y) {
  %1 = call i16 @llvm.nvvm.mulhi.s(i16 %x, i16 %y)
  ret i16 %1
}

; CHECK-LABEL: test_mulhi_u16
; CHECK: mul.hi.u16
define i16 @test_mulhi_u16(i16 %x, i16 %y) {
  %1 = call i16 @llvm.nvvm.mulhi.us(i16 %x, i16 %y)
  ret i16 %1
}

; CHECK-LABEL: test_mulhi_i32
; CHECK: mul.hi.s32
define i32 @test_mulhi_i32(i32 %x, i32 %y) {
  %1 = call i32 @llvm.nvvm.mulhi.i(i32 %x, i32 %y)
  ret i32 %1
}

; CHECK-LABEL: test_mulhi_u32
; CHECK: mul.hi.u32
define i32 @test_mulhi_u32(i32 %x, i32 %y) {
  %1 = call i32 @llvm.nvvm.mulhi.ui(i32 %x, i32 %y)
  ret i32 %1
}

; CHECK-LABEL: test_mulhi_i64
; CHECK: mul.hi.s64
define i64 @test_mulhi_i64(i64 %x, i64 %y) {
  %1 = call i64 @llvm.nvvm.mulhi.ll(i64 %x, i64 %y)
  ret i64 %1
}

; CHECK-LABEL: test_mulhi_u64
; CHECK: mul.hi.u64
define i64 @test_mulhi_u64(i64 %x, i64 %y) {
  %1 = call i64 @llvm.nvvm.mulhi.ull(i64 %x, i64 %y)
  ret i64 %1
}

declare i16 @llvm.nvvm.mulhi.s(i16, i16)
declare i16 @llvm.nvvm.mulhi.us(i16, i16)
declare i32 @llvm.nvvm.mulhi.i(i32, i32)
declare i32 @llvm.nvvm.mulhi.ui(i32, i32)
declare i64 @llvm.nvvm.mulhi.ll(i64, i64)
declare i64 @llvm.nvvm.mulhi.ull(i64, i64)
