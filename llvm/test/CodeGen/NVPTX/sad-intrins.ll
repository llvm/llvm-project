; RUN: llc < %s -march=nvptx -mcpu=sm_50 | FileCheck %s

; CHECK-LABEL: test_sad_i16
; CHECK: sad.s16
define i16 @test_sad_i16(i16 %x, i16 %y, i16 %z) {
  %1 = call i16 @llvm.nvvm.sad.s(i16 %x, i16 %y, i16 %z)
  ret i16 %1
}

; CHECK-LABEL: test_sad_u16
; CHECK: sad.u16
define i16 @test_sad_u16(i16 %x, i16 %y, i16 %z) {
  %1 = call i16 @llvm.nvvm.sad.us(i16 %x, i16 %y, i16 %z)
  ret i16 %1
}

; CHECK-LABEL: test_sad_i32
; CHECK: sad.s32
define i32 @test_sad_i32(i32 %x, i32 %y, i32 %z) {
  %1 = call i32 @llvm.nvvm.sad.i(i32 %x, i32 %y, i32 %z)
  ret i32 %1
}

; CHECK-LABEL: test_sad_u32
; CHECK: sad.u32
define i32 @test_sad_u32(i32 %x, i32 %y, i32 %z) {
  %1 = call i32 @llvm.nvvm.sad.ui(i32 %x, i32 %y, i32 %z)
  ret i32 %1
}

; CHECK-LABEL: test_sad_i64
; CHECK: sad.s64
define i64 @test_sad_i64(i64 %x, i64 %y, i64 %z) {
  %1 = call i64 @llvm.nvvm.sad.ll(i64 %x, i64 %y, i64 %z)
  ret i64 %1
}

; CHECK-LABEL: test_sad_u64
; CHECK: sad.u64
define i64 @test_sad_u64(i64 %x, i64 %y, i64 %z) {
  %1 = call i64 @llvm.nvvm.sad.ull(i64 %x, i64 %y, i64 %z)
  ret i64 %1
}

declare i16 @llvm.nvvm.sad.s(i16, i16, i16)
declare i16 @llvm.nvvm.sad.us(i16, i16, i16)
declare i32 @llvm.nvvm.sad.i(i32, i32, i32)
declare i32 @llvm.nvvm.sad.ui(i32, i32, i32)
declare i64 @llvm.nvvm.sad.ll(i64, i64, i64)
declare i64 @llvm.nvvm.sad.ull(i64, i64, i64)
