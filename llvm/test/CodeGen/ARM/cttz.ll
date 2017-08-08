; RUN: llc < %s -mtriple arm-eabi -mattr=+v6t2 | FileCheck %s
; RUN: llc < %s -mtriple arm-eabi -mattr=+v6t2 -mattr=+neon | FileCheck %s

; This test checks the @llvm.cttz.* intrinsics for integers.

declare i8 @llvm.cttz.i8(i8, i1)
declare i16 @llvm.cttz.i16(i16, i1)
declare i32 @llvm.cttz.i32(i32, i1)
declare i64 @llvm.cttz.i64(i64, i1)

;------------------------------------------------------------------------------

define i8 @test_i8(i8 %a) {
; CHECK-LABEL: test_i8:
; CHECK: orr [[REG:r[0-9]+]], [[REG]], #256
; CHECK: rbit
; CHECK: clz
  %tmp = call i8 @llvm.cttz.i8(i8 %a, i1 false)
  ret i8 %tmp
}

define i16 @test_i16(i16 %a) {
; CHECK-LABEL: test_i16:
; CHECK: orr [[REG:r[0-9]+]], [[REG]], #65536
; CHECK: rbit
; CHECK: clz
  %tmp = call i16 @llvm.cttz.i16(i16 %a, i1 false)
  ret i16 %tmp
}

define i32 @test_i32(i32 %a) {
; CHECK-LABEL: test_i32:
; CHECK: rbit
; CHECK: clz
  %tmp = call i32 @llvm.cttz.i32(i32 %a, i1 false)
  ret i32 %tmp
}

define i64 @test_i64(i64 %a) {
; CHECK-LABEL: test_i64:
; CHECK: rbit
; CHECK: rbit
; CHECK: clz
; CHECK: cmp
; CHECK: add
; CHECK: clzne
  %tmp = call i64 @llvm.cttz.i64(i64 %a, i1 false)
  ret i64 %tmp
}

;------------------------------------------------------------------------------

define i8 @test_i8_zero_undef(i8 %a) {
; CHECK-LABEL: test_i8_zero_undef:
; CHECK-NOT: orr
; CHECK: rbit
; CHECK: clz
  %tmp = call i8 @llvm.cttz.i8(i8 %a, i1 true)
  ret i8 %tmp
}

define i16 @test_i16_zero_undef(i16 %a) {
; CHECK-LABEL: test_i16_zero_undef:
; CHECK-NOT: orr
; CHECK: rbit
; CHECK: clz
  %tmp = call i16 @llvm.cttz.i16(i16 %a, i1 true)
  ret i16 %tmp
}


define i32 @test_i32_zero_undef(i32 %a) {
; CHECK-LABEL: test_i32_zero_undef:
; CHECK: rbit
; CHECK: clz
  %tmp = call i32 @llvm.cttz.i32(i32 %a, i1 true)
  ret i32 %tmp
}

define i64 @test_i64_zero_undef(i64 %a) {
; CHECK-LABEL: test_i64_zero_undef:
; CHECK: rbit
; CHECK: rbit
; CHECK: clz
; CHECK: cmp
; CHECK: add
; CHECK: clzne
  %tmp = call i64 @llvm.cttz.i64(i64 %a, i1 true)
  ret i64 %tmp
}
