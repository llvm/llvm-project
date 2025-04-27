; RUN: opt -passes=instcombine -S < %s | FileCheck %s

declare i32 @llvm.cttz.i32(i32, i1)

define i32 @test_cttz_lshr(i32 %x) {
; CHECK-LABEL: @test_cttz_lshr(
; CHECK: call range(i32 0, 33) i32 @llvm.cttz.i32(
; CHECK: lshr exact i32
; CHECK: ret i32
  %cttz = call i32 @llvm.cttz.i32(i32 %x, i1 false)
  %sh = lshr i32 %x, %cttz
  ret i32 %sh
}

define i32 @test_cttz_ashr(i32 %x) {
; CHECK-LABEL: @test_cttz_ashr(
; CHECK: call range(i32 0, 33) i32 @llvm.cttz.i32(
; CHECK: ashr exact i32
; CHECK: ret i32
  %cttz = call i32 @llvm.cttz.i32(i32 %x, i1 true)
  %sh = ashr i32 %x, %cttz
  ret i32 %sh
}

