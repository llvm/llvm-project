; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv1.6-vulkan1.3-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-vulkan1.3-unknown %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}


; CHECK-DAG: [[i8_t:%.+]]  = OpTypeInt 8 0
; CHECK-DAG: [[i16_t:%.+]] = OpTypeInt 16 0
; CHECK-DAG: [[i32_t:%.+]] = OpTypeInt 32 0
; CHECK-DAG: [[i64_t:%.+]] = OpTypeInt 64 0
; CHECK-DAG: [[i32x2_t:%.+]] = OpTypeVector [[i32_t]] 2
; CHECK-DAG: [[i32x3_t:%.+]] = OpTypeVector [[i32_t]] 3
; CHECK-DAG: [[i32x4_t:%.+]] = OpTypeVector [[i32_t]] 4
; CHECK-DAG: [[i64x2_t:%.+]] = OpTypeVector [[i64_t]] 2
; CHECK-DAG: [[i64x3_t:%.+]] = OpTypeVector [[i64_t]] 3
; CHECK-DAG: [[i64x4_t:%.+]] = OpTypeVector [[i64_t]] 4
; CHECK-DAG: [[i16x3_t:%.+]] = OpTypeVector [[i16_t]] 3

; CHECK-DAG: [[i64_32:%.+]] = OpConstant [[i64_t]] 32
; CHECK-DAG: [[i64x2_32:%.+]] = OpConstantComposite [[i64x2_t]] [[i64_32]] [[i64_32]]
; CHECK-DAG: [[i64x3_32:%.+]] = OpConstantComposite [[i64x3_t]] [[i64_32]] [[i64_32]] [[i64_32]]
; CHECK-DAG: [[i64x4_32:%.+]] = OpConstantComposite [[i64x4_t]] [[i64_32]] [[i64_32]] [[i64_32]] [[i64_32]]

; CHECK-LABEL:  ; -- Begin function test

; CHECK: [[p8:%.+]] = OpFunctionParameter [[i8_t]]
; CHECK: [[p16:%.+]] = OpFunctionParameter [[i16_t]]
; CHECK: [[p32:%.+]] = OpFunctionParameter [[i32_t]]
; CHECK: [[p64:%.+]] = OpFunctionParameter [[i64_t]]
; CHECK: [[p32x2:%.+]] = OpFunctionParameter [[i32x2_t]]
; CHECK: [[p64x2:%.+]] = OpFunctionParameter [[i64x2_t]]
; CHECK: [[p64x3:%.+]] = OpFunctionParameter [[i64x3_t]]
; CHECK: [[p64x4:%.+]] = OpFunctionParameter [[i64x4_t]]
; CHECK: [[p16x3:%.+]] = OpFunctionParameter [[i16x3_t]]

; p8
; CHECK: [[p8_conversion_in:%.+]] = OpUConvert [[i32_t]] [[p8]]
; CHECK: [[p8_bitcount:%.+]] = OpBitCount [[i32_t]] [[p8_conversion_in]]
; CHECK: %[[#]] = OpUConvert [[i8_t]] [[p8_bitcount]]

; p16
; CHECK: [[p16_conversion_in:%.+]] = OpUConvert [[i32_t]] [[p16]]
; CHECK: [[p16_bitcount:%.+]] = OpBitCount [[i32_t]] [[p16_conversion_in]]
; CHECK: %[[#]] = OpUConvert [[i16_t]] [[p16_bitcount]]

; p32
; CHECK: [[p32_bitcount:%.+]] = OpBitCount [[i32_t]] [[p32]]

; p64
; CHECK: [[p64_trunc_low:%.+]] = OpUConvert [[i32_t]] [[p64]]
; CHECK: [[p64_low:%.+]] = OpBitCount [[i32_t]] [[p64_trunc_low]]
; CHECK: [[p64_shift_high:%.+]] = OpShiftRightLogical [[i64_t]] [[p64]] [[i64_32]]
; CHECK: [[p64_trunc_high:%.+]] = OpUConvert [[i32_t]] [[p64_shift_high]]
; CHECK: [[p64_high:%.+]] = OpBitCount [[i32_t]] [[p64_trunc_high]]
; CHECK: [[p64_sum:%.+]] = OpIAdd [[i32_t]] [[p64_high]] [[p64_low]]
; CHECK: %[[#]] = OpUConvert [[i64_t]] [[p64_sum]]

; p32x2
; CHECK: [[#]] = OpBitCount [[i32x2_t]] [[p32x2]]

; p64x2
; CHECK: [[p64x2_trunc_low:%.+]] = OpUConvert [[i32x2_t]] [[p64x2]]
; CHECK: [[p64x2_low:%.+]] = OpBitCount [[i32x2_t]] [[p64x2_trunc_low]]
; CHECK: [[p64x2_shift_high:%.+]] = OpShiftRightLogical [[i64x2_t]] [[p64x2]] [[i64x2_32]]
; CHECK: [[p64x2_trunc_high:%.+]] = OpUConvert [[i32x2_t]] [[p64x2_shift_high]]
; CHECK: [[p64x2_high:%.+]] = OpBitCount [[i32x2_t]] [[p64x2_trunc_high]]
; CHECK: [[p64x2_sum:%.+]] = OpIAdd [[i32x2_t]] [[p64x2_high]] [[p64x2_low]]
; CHECK: %[[#]] = OpUConvert [[i64x2_t]] [[p64x2_sum]]

; p64x3
; CHECK: [[p64x3_trunc_low:%.+]] = OpUConvert [[i32x3_t]] [[p64x3]]
; CHECK: [[p64x3_low:%.+]] = OpBitCount [[i32x3_t]] [[p64x3_trunc_low]]
; CHECK: [[p64x3_shift_high:%.+]] = OpShiftRightLogical [[i64x3_t]] [[p64x3]] [[i64x3_32]]
; CHECK: [[p64x3_trunc_high:%.+]] = OpUConvert [[i32x3_t]] [[p64x3_shift_high]]
; CHECK: [[p64x3_high:%.+]] = OpBitCount [[i32x3_t]] [[p64x3_trunc_high]]
; CHECK: [[p64x3_sum:%.+]] = OpIAdd [[i32x3_t]] [[p64x3_high]] [[p64x3_low]]
; CHECK: %[[#]] = OpUConvert [[i64x3_t]] [[p64x3_sum]]

; p64x4
; CHECK: [[p64x4_trunc_low:%.+]] = OpUConvert [[i32x4_t]] [[p64x4]]
; CHECK: [[p64x4_low:%.+]] = OpBitCount [[i32x4_t]] [[p64x4_trunc_low]]
; CHECK: [[p64x4_shift_high:%.+]] = OpShiftRightLogical [[i64x4_t]] [[p64x4]] [[i64x4_32]]
; CHECK: [[p64x4_trunc_high:%.+]] = OpUConvert [[i32x4_t]] [[p64x4_shift_high]]
; CHECK: [[p64x4_high:%.+]] = OpBitCount [[i32x4_t]] [[p64x4_trunc_high]]
; CHECK: [[p64x4_sum:%.+]] = OpIAdd [[i32x4_t]] [[p64x4_high]] [[p64x4_low]]
; CHECK: %[[#]] = OpUConvert [[i64x4_t]] [[p64x4_sum]]

; p16x3
; CHECK: [[p16_conversion_in:%.+]] = OpUConvert [[i32x3_t]] [[p16x3]]
; CHECK: [[p16_bitcount:%.+]] = OpBitCount [[i32x3_t]] [[p16_conversion_in]]
; CHECK: %[[#]] = OpUConvert [[i16x3_t]] [[p16_bitcount]]

@g1 = private global i8  0, align 4
@g2 = private global i16 0, align 4
@g3 = private global i32 0, align 4
@g4 = private global i64 0, align 8
@g5 = private global <2 x i32> zeroinitializer, align 4
@g6 = private global <2 x i64> zeroinitializer, align 8
@g7 = private global <3 x i64> zeroinitializer, align 8
@g8 = private global <4 x i64> zeroinitializer, align 8
@g9 = private global <3 x i16> zeroinitializer, align 4


define internal void @test(i8 %x8, i16 %x16, i32 %x32, i64 %x64, <2 x i32> %x2i32, <2 x i64> %x2i64, <3 x i64> %x3i64, <4 x i64> %x4i64, <3 x i16> %x3i16) local_unnamed_addr {
entry:
  %0 = tail call i8 @llvm.ctpop.i8(i8 %x8)
  store i8 %0, ptr @g1, align 4
  %1 = tail call i16 @llvm.ctpop.i16(i16 %x16)
  store i16 %1, ptr @g2, align 4
  %2 = tail call i32 @llvm.ctpop.i32(i32 %x32)
  store i32 %2, ptr @g3, align 4
  %3 = tail call i64 @llvm.ctpop.i64(i64 %x64)
  store i64 %3, ptr @g4, align 8
  %4 = tail call <2 x i32> @llvm.ctpop.v2i32(<2 x i32> %x2i32)
  store <2 x i32> %4, ptr @g5, align 4
  %5 = tail call <2 x i64> @llvm.ctpop.v2i64(<2 x i64> %x2i64)
  store <2 x i64> %5, ptr @g6, align 4
  %6 = tail call <3 x i64> @llvm.ctpop.v3i64(<3 x i64> %x3i64)
  store <3 x i64> %6, ptr @g7, align 4
  %7 = tail call <4 x i64> @llvm.ctpop.v4i64(<4 x i64> %x4i64)
  store <4 x i64> %7, ptr @g8, align 4
  %8 = tail call <3 x i16> @llvm.ctpop.v3i16(<3 x i16> %x3i16)
  store <3 x i16> %8, ptr @g9, align 4
  ret void
}

define void @main() #1 {
entry:
  ret void
}

attributes #1 = { convergent noinline norecurse "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
