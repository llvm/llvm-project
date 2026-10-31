; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv1.6-vulkan1.3-unknown %s -o - | FileCheck %s --check-prefix=CHECK,CHECK-SCALAR
; RUN: llc -spirv-ext=+SPV_EXT_long_vector -verify-machineinstrs -O0 -mtriple=spirv1.6-vulkan1.3-unknown %s -o - | FileCheck --check-prefix=CHECK,CHECK-VECTOR %s
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

; CHECK-VECTOR-DAG: [[i32_one:%.+]] = OpConstant [[i32_t]] 1
; CHECK-VECTOR-DAG: [[i16x1_t:%.+]] = OpTypeVectorIdEXT [[i16_t]] [[i32_one]]
; CHECK-VECTOR-DAG: [[i32x1_t:%.+]] = OpTypeVectorIdEXT [[i32_t]] [[i32_one]]
; CHECK-VECTOR-DAG: [[i64x1_t:%.+]] = OpTypeVectorIdEXT [[i64_t]] [[i32_one]]

; CHECK-DAG: [[i64_32:%.+]] = OpConstant [[i64_t]] 32
; CHECK-VECTOR-DAG: [[i64x1_32:%.+]] = OpConstantComposite [[i64x1_t]] [[i64_32]]
; CHECK-DAG: [[i64x2_32:%.+]] = OpConstantComposite [[i64x2_t]] [[i64_32]] [[i64_32]]
; CHECK-DAG: [[i64x3_32:%.+]] = OpConstantComposite [[i64x3_t]] [[i64_32]] [[i64_32]] [[i64_32]]
; CHECK-DAG: [[i64x4_32:%.+]] = OpConstantComposite [[i64x4_t]] [[i64_32]] [[i64_32]] [[i64_32]] [[i64_32]]

@g8 = private global i8  0, align 4
@g16 = private global i16 0, align 4
@g32 = private global i32 0, align 4
@g64 = private global i64 0, align 8
@g2i32 = private global <2 x i32> zeroinitializer, align 4
@g2i64 = private global <2 x i64> zeroinitializer, align 8
@g3i64 = private global <3 x i64> zeroinitializer, align 8
@g4i64 = private global <4 x i64> zeroinitializer, align 8
@g3i16 = private global <3 x i16> zeroinitializer, align 4
@g1i16 = private global <1 x i16> zeroinitializer, align 4
@g1i32 = private global <1 x i32> zeroinitializer, align 4
@g1i64 = private global <1 x i64> zeroinitializer, align 8

define internal void @test(i8 %x8, i16 %x16, i32 %x32, i64 %x64, <2 x i32> %x2i32, <2 x i64> %x2i64, <3 x i64> %x3i64, <4 x i64> %x4i64, <3 x i16> %x3i16, <1 x i16> %x1i16, <1 x i32> %x1i32, <1 x i64> %x1i64) local_unnamed_addr {
entry:
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
  ; CHECK-SCALAR: [[p16x1:%.+]] = OpFunctionParameter [[i16_t]]
  ; CHECK-VECTOR: [[p16x1:%.+]] = OpFunctionParameter [[i16x1_t]]
  ; CHECK-SCALAR: [[p32x1:%.+]] = OpFunctionParameter [[i32_t]]
  ; CHECK-VECTOR: [[p32x1:%.+]] = OpFunctionParameter [[i32x1_t]]
  ; CHECK-SCALAR: [[p64x1:%.+]] = OpFunctionParameter [[i64_t]]
  ; CHECK-VECTOR: [[p64x1:%.+]] = OpFunctionParameter [[i64x1_t]]

  ; p8
  ; CHECK: [[p8_conversion_in:%.+]] = OpUConvert [[i32_t]] [[p8]]
  ; CHECK: [[p8_bitcount:%.+]] = OpBitCount [[i32_t]] [[p8_conversion_in]]
  ; CHECK: %[[#]] = OpUConvert [[i8_t]] [[p8_bitcount]]
  %y8 = tail call i8 @llvm.ctpop.i8(i8 %x8)
  store i8 %y8, ptr @g8, align 4

  ; p16
  ; CHECK: [[p16_conversion_in:%.+]] = OpUConvert [[i32_t]] [[p16]]
  ; CHECK: [[p16_bitcount:%.+]] = OpBitCount [[i32_t]] [[p16_conversion_in]]
  ; CHECK: %[[#]] = OpUConvert [[i16_t]] [[p16_bitcount]]
  %y16 = tail call i16 @llvm.ctpop.i16(i16 %x16)
  store i16 %y16, ptr @g16, align 4

  ; p32
  ; CHECK: [[p32_bitcount:%.+]] = OpBitCount [[i32_t]] [[p32]]
  %y32 = tail call i32 @llvm.ctpop.i32(i32 %x32)
  store i32 %y32, ptr @g32, align 4

  ; p64
  ; CHECK: [[p64_trunc_low:%.+]] = OpUConvert [[i32_t]] [[p64]]
  ; CHECK: [[p64_low:%.+]] = OpBitCount [[i32_t]] [[p64_trunc_low]]
  ; CHECK: [[p64_shift_high:%.+]] = OpShiftRightLogical [[i64_t]] [[p64]] [[i64_32]]
  ; CHECK: [[p64_trunc_high:%.+]] = OpUConvert [[i32_t]] [[p64_shift_high]]
  ; CHECK: [[p64_high:%.+]] = OpBitCount [[i32_t]] [[p64_trunc_high]]
  ; CHECK: [[p64_sum:%.+]] = OpIAdd [[i32_t]] [[p64_high]] [[p64_low]]
  ; CHECK: %[[#]] = OpUConvert [[i64_t]] [[p64_sum]]
  %y64 = tail call i64 @llvm.ctpop.i64(i64 %x64)
  store i64 %y64, ptr @g64, align 8

  ; p32x2
  ; CHECK: [[#]] = OpBitCount [[i32x2_t]] [[p32x2]]
  %y2i32 = tail call <2 x i32> @llvm.ctpop.v2i32(<2 x i32> %x2i32)
  store <2 x i32> %y2i32, ptr @g2i32, align 4

  ; p64x2
  ; CHECK: [[p64x2_trunc_low:%.+]] = OpUConvert [[i32x2_t]] [[p64x2]]
  ; CHECK: [[p64x2_low:%.+]] = OpBitCount [[i32x2_t]] [[p64x2_trunc_low]]
  ; CHECK: [[p64x2_shift_high:%.+]] = OpShiftRightLogical [[i64x2_t]] [[p64x2]] [[i64x2_32]]
  ; CHECK: [[p64x2_trunc_high:%.+]] = OpUConvert [[i32x2_t]] [[p64x2_shift_high]]
  ; CHECK: [[p64x2_high:%.+]] = OpBitCount [[i32x2_t]] [[p64x2_trunc_high]]
  ; CHECK: [[p64x2_sum:%.+]] = OpIAdd [[i32x2_t]] [[p64x2_high]] [[p64x2_low]]
  ; CHECK: %[[#]] = OpUConvert [[i64x2_t]] [[p64x2_sum]]
  %y2i64 = tail call <2 x i64> @llvm.ctpop.v2i64(<2 x i64> %x2i64)
  store <2 x i64> %y2i64, ptr @g2i64, align 4

  ; p64x3
  ; CHECK: [[p64x3_trunc_low:%.+]] = OpUConvert [[i32x3_t]] [[p64x3]]
  ; CHECK: [[p64x3_low:%.+]] = OpBitCount [[i32x3_t]] [[p64x3_trunc_low]]
  ; CHECK: [[p64x3_shift_high:%.+]] = OpShiftRightLogical [[i64x3_t]] [[p64x3]] [[i64x3_32]]
  ; CHECK: [[p64x3_trunc_high:%.+]] = OpUConvert [[i32x3_t]] [[p64x3_shift_high]]
  ; CHECK: [[p64x3_high:%.+]] = OpBitCount [[i32x3_t]] [[p64x3_trunc_high]]
  ; CHECK: [[p64x3_sum:%.+]] = OpIAdd [[i32x3_t]] [[p64x3_high]] [[p64x3_low]]
  ; CHECK: %[[#]] = OpUConvert [[i64x3_t]] [[p64x3_sum]]
  %y3i64 = tail call <3 x i64> @llvm.ctpop.v3i64(<3 x i64> %x3i64)
  store <3 x i64> %y3i64, ptr @g3i64, align 4

  ; p64x4
  ; CHECK: [[p64x4_trunc_low:%.+]] = OpUConvert [[i32x4_t]] [[p64x4]]
  ; CHECK: [[p64x4_low:%.+]] = OpBitCount [[i32x4_t]] [[p64x4_trunc_low]]
  ; CHECK: [[p64x4_shift_high:%.+]] = OpShiftRightLogical [[i64x4_t]] [[p64x4]] [[i64x4_32]]
  ; CHECK: [[p64x4_trunc_high:%.+]] = OpUConvert [[i32x4_t]] [[p64x4_shift_high]]
  ; CHECK: [[p64x4_high:%.+]] = OpBitCount [[i32x4_t]] [[p64x4_trunc_high]]
  ; CHECK: [[p64x4_sum:%.+]] = OpIAdd [[i32x4_t]] [[p64x4_high]] [[p64x4_low]]
  ; CHECK: %[[#]] = OpUConvert [[i64x4_t]] [[p64x4_sum]]
  %y4i64 = tail call <4 x i64> @llvm.ctpop.v4i64(<4 x i64> %x4i64)
  store <4 x i64> %y4i64, ptr @g4i64, align 4

  ; p16x3
  ; CHECK: [[p16_conversion_in:%.+]] = OpUConvert [[i32x3_t]] [[p16x3]]
  ; CHECK: [[p16_bitcount:%.+]] = OpBitCount [[i32x3_t]] [[p16_conversion_in]]
  ; CHECK: %[[#]] = OpUConvert [[i16x3_t]] [[p16_bitcount]]
  %y3i16 = tail call <3 x i16> @llvm.ctpop.v3i16(<3 x i16> %x3i16)
  store <3 x i16> %y3i16, ptr @g3i16, align 4

  ; p16x1
  ;
  ; CHECK-SCALAR: [[p16x1_conversion_in:%.+]] = OpUConvert [[i32_t]] [[p16x1]]
  ; CHECK-SCALAR: [[p16x1_bitcount:%.+]] = OpBitCount [[i32_t]] [[p16x1_conversion_in]]
  ; CHECK-SCALAR: %[[#]] = OpUConvert [[i16_t]] [[p16x1_bitcount]]
  ;
  ; CHECK-VECTOR: [[p16x1_conversion_in:%.+]] = OpUConvert [[i32x1_t]] [[p16x1]]
  ; CHECK-VECTOR: [[p16x1_bitcount:%.+]] = OpBitCount [[i32x1_t]] [[p16x1_conversion_in]]
  ; CHECK-VECTOR: %[[#]] = OpUConvert [[i16x1_t]] [[p16x1_bitcount]]
  %y1i16 = tail call <1 x i16> @llvm.ctpop.v1i16(<1 x i16> %x1i16)
  store <1 x i16> %y1i16, ptr @g1i16, align 4

  ; p32x1
  ; CHECK-SCALAR: [[p32x1_bitcount:%.+]] = OpBitCount [[i32_t]] [[p32x1]]
  ; CHECK-VECTOR: [[p32x1_bitcount:%.+]] = OpBitCount [[i32x1_t]] [[p32x1]]
  %y1i32 = tail call <1 x i32> @llvm.ctpop.v1i32(<1 x i32> %x1i32)
  store <1 x i32> %y1i32, ptr @g1i32, align 4

  ; p64x1
  ; CHECK-SCALAR: [[p64x1_trunc_low:%.+]] = OpUConvert [[i32_t]] [[p64x1]]
  ; CHECK-SCALAR: [[p64x1_low:%.+]] = OpBitCount [[i32_t]] [[p64x1_trunc_low]]
  ; CHECK-SCALAR: [[p64x1_shift_high:%.+]] = OpShiftRightLogical [[i64_t]] [[p64x1]] [[i64_32]]
  ; CHECK-SCALAR: [[p64x1_trunc_high:%.+]] = OpUConvert [[i32_t]] [[p64x1_shift_high]]
  ; CHECK-SCALAR: [[p64x1_high:%.+]] = OpBitCount [[i32_t]] [[p64x1_trunc_high]]
  ; CHECK-SCALAR: [[p64x1_sum:%.+]] = OpIAdd [[i32_t]] [[p64x1_high]] [[p64x1_low]]
  ; CHECK-SCALAR: %[[#]] = OpUConvert [[i64_t]] [[p64x1_sum]]
  ;
  ; CHECK-VECTOR: [[p64x1_trunc_low:%.+]] = OpUConvert [[i32x1_t]] [[p64x1]]
  ; CHECK-VECTOR: [[p64x1_low:%.+]] = OpBitCount [[i32x1_t]] [[p64x1_trunc_low]]
  ; CHECK-VECTOR: [[p64x1_shift_high:%.+]] = OpShiftRightLogical [[i64x1_t]] [[p64x1]] [[i64x1_32]]
  ; CHECK-VECTOR: [[p64x1_trunc_high:%.+]] = OpUConvert [[i32x1_t]] [[p64x1_shift_high]]
  ; CHECK-VECTOR: [[p64x1_high:%.+]] = OpBitCount [[i32x1_t]] [[p64x1_trunc_high]]
  ; CHECK-VECTOR: [[p64x1_sum:%.+]] = OpIAdd [[i32x1_t]] [[p64x1_high]] [[p64x1_low]]
  ; CHECK-VECTOR: %[[#]] = OpUConvert [[i64x1_t]] [[p64x1_sum]]
  %y1i64 = tail call <1 x i64> @llvm.ctpop.v1i64(<1 x i64> %x1i64)
  store <1 x i64> %y1i64, ptr @g1i64, align 8
  ret void
}

define void @main() #1 {
entry:
  ret void
}

attributes #1 = { convergent noinline norecurse "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
