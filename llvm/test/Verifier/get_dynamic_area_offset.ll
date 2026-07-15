; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

target datalayout = "p0:64:64-p5:32:32-A5"

declare i64 @llvm.get.dynamic.area.offset.i64()

; CHECK: get_dynamic_area_offset result type must be scalar integer matching alloca address space width
; CHECK-NEXT: %res = call i64 @llvm.get.dynamic.area.offset.i64()
define i64 @test_dynamic_area_too_big() {
  %res = call i64 @llvm.get.dynamic.area.offset.i64()
  ret i64 %res
}

declare i16 @llvm.get.dynamic.area.offset.i16()

; CHECK: get_dynamic_area_offset result type must be scalar integer matching alloca address space width
; CHECK-NEXT: %res = call i16 @llvm.get.dynamic.area.offset.i16()
define i16 @test_dynamic_area_too_small() {
  %res = call i16 @llvm.get.dynamic.area.offset.i16()
  ret i16 %res
}

declare <2 x i32> @llvm.get.dynamic.area.offset.v2i32()

; CHECK: get_dynamic_area_offset result type must be scalar integer matching alloca address space width
; CHECK-NEXT: %res = call <2 x i32> @llvm.get.dynamic.area.offset.v2i32()
define <2 x i32> @test_dynamic_area_vector() {
  %res = call <2 x i32> @llvm.get.dynamic.area.offset.v2i32()
  ret <2 x i32> %res
}
