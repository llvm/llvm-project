; RUN: opt -S -passes=spirv-legalize-zero-size-arrays -mtriple=spirv64-unknown-unknown < %s | FileCheck %s
; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=SPIRV --implicit-check-not=OpTypeArray --implicit-check-not=OpTypeRuntimeArray --implicit-check-not=VariableLengthArrayINTEL
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; SPIRV: %[[#BYTE:]] = OpTypeInt 8 0
; SPIRV: %[[#PTR:]] = OpTypePointer Function %[[#BYTE]]

; Test that zero-size array alloca with dynamic count allocates just i8

define void @test_alloca_with_count(i32 %n) {
; CHECK-LABEL: @test_alloca_with_count(
; CHECK-NEXT:    [[ARR:%.*]] = alloca i8, align 4
; CHECK-NEXT:    ret void
  %arr = alloca [0 x i32], i32 %n, align 4
  ret void
}

; SPIRV-LABEL: Begin function test_alloca_zero_count
; SPIRV: OpVariable %[[#PTR]] Function
define i64 @test_alloca_zero_count() {
; CHECK-LABEL: @test_alloca_zero_count(
; CHECK-NEXT:    [[ARR:%.*]] = alloca i8, align 16
; CHECK-NEXT:    [[ADDR:%.*]] = ptrtoint ptr [[ARR]] to i64
; CHECK-NEXT:    ret i64 [[ADDR]]
  %arr = alloca i32, i32 0, align 16
  %addr = ptrtoint ptr %arr to i64
  ret i64 %addr
}
