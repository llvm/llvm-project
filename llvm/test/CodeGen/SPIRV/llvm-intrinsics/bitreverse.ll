; Test bitreverse for all popular types (scalars and vectors of 8, 16, 32, 64 bits)
; This comprehensive test covers the full range of supported types.

; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-EMULATION
; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_bit_instructions %s -o - | FileCheck %s --check-prefix=CHECK-NATIVE
; RUN: %if spirv-tools %{ llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_bit_instructions %s -o - -filetype=obj | spirv-val %}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; Scalar types - 8, 16, 32, 64 bits
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; CHECK-EMULATION-NOT: OpExtension "SPV_KHR_bit_instructions"
; CHECK-NATIVE-DAG: OpExtension "SPV_KHR_bit_instructions"


; CHECK-NATIVE-DAG: %[[#I8:]] = OpTypeInt 8 0
; CHECK-NATIVE-DAG: %[[#I16:]] = OpTypeInt 16 0
; CHECK-NATIVE-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-NATIVE-DAG: %[[#I64:]] = OpTypeInt 64 0

; CHECK-NATIVE-DAG: %[[#V2I8:]] = OpTypeVector %[[#I8]] 2
; CHECK-NATIVE-DAG: %[[#V2I16:]] = OpTypeVector %[[#I16]] 2
; CHECK-NATIVE-DAG: %[[#V2I32:]] = OpTypeVector %[[#I32]] 2
; CHECK-NATIVE-DAG: %[[#V2I64:]] = OpTypeVector %[[#I64]] 2

; CHECK-NATIVE-DAG: %[[#V4I8:]] = OpTypeVector %[[#I8]] 4
; CHECK-NATIVE-DAG: %[[#V4I16:]] = OpTypeVector %[[#I16]] 4
; CHECK-NATIVE-DAG: %[[#V4I32:]] = OpTypeVector %[[#I32]] 4
; CHECK-NATIVE-DAG: %[[#V4I64:]] = OpTypeVector %[[#I64]] 4

; CHECK-NATIVE-DAG: %[[#V8I8:]] = OpTypeVector %[[#I8]] 8
; CHECK-NATIVE-DAG: %[[#V8I16:]] = OpTypeVector %[[#I16]] 8
; CHECK-NATIVE-DAG: %[[#V8I32:]] = OpTypeVector %[[#I32]] 8
; CHECK-NATIVE-DAG: %[[#V8I64:]] = OpTypeVector %[[#I64]] 8

; CHECK-NATIVE-DAG: %[[#V16I8:]] = OpTypeVector %[[#I8]] 16
; CHECK-NATIVE-DAG: %[[#V16I16:]] = OpTypeVector %[[#I16]] 16
; CHECK-NATIVE-DAG: %[[#V16I32:]] = OpTypeVector %[[#I32]] 16
; CHECK-NATIVE-DAG: %[[#V16I64:]] = OpTypeVector %[[#I64]] 16



; CHECK-EMULATION-DAG: %[[#I8:]] = OpTypeInt 8 0
; CHECK-EMULATION-DAG: %[[#I16:]] = OpTypeInt 16 0
; CHECK-EMULATION-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-EMULATION-DAG: %[[#I64:]] = OpTypeInt 64 0

; CHECK-EMULATION-DAG: %[[#V2I8:]] = OpTypeVector %[[#I8]] 2
; CHECK-EMULATION-DAG: %[[#V2I16:]] = OpTypeVector %[[#I16]] 2
; CHECK-EMULATION-DAG: %[[#V2I32:]] = OpTypeVector %[[#I32]] 2
; CHECK-EMULATION-DAG: %[[#V2I64:]] = OpTypeVector %[[#I64]] 2

; CHECK-EMULATION-DAG: %[[#V4I8:]] = OpTypeVector %[[#I8]] 4
; CHECK-EMULATION-DAG: %[[#V4I16:]] = OpTypeVector %[[#I16]] 4
; CHECK-EMULATION-DAG: %[[#V4I32:]] = OpTypeVector %[[#I32]] 4
; CHECK-EMULATION-DAG: %[[#V4I64:]] = OpTypeVector %[[#I64]] 4

; CHECK-EMULATION-DAG: %[[#V8I8:]] = OpTypeVector %[[#I8]] 8
; CHECK-EMULATION-DAG: %[[#V8I16:]] = OpTypeVector %[[#I16]] 8
; CHECK-EMULATION-DAG: %[[#V8I32:]] = OpTypeVector %[[#I32]] 8
; CHECK-EMULATION-DAG: %[[#V8I64:]] = OpTypeVector %[[#I64]] 8

; CHECK-EMULATION-DAG: %[[#V16I8:]] = OpTypeVector %[[#I8]] 16
; CHECK-EMULATION-DAG: %[[#V16I16:]] = OpTypeVector %[[#I16]] 16
; CHECK-EMULATION-DAG: %[[#V16I32:]] = OpTypeVector %[[#I32]] 16
; CHECK-EMULATION-DAG: %[[#V16I64:]] = OpTypeVector %[[#I64]] 16

define spir_kernel void @test_scalar(i8 %a8, i16 %a16, i32 %a32, i64 %a64, ptr addrspace(1) %res) #0 {
entry:
  ; CHECK-NATIVE-LABEL: Begin function test_scalar
  ; CHECK-EMULATION-LABEL: Begin function test_scalar
  ; CHECK-EMULATION-NOT: OpBitReverse

  ; CHECK-NATIVE-DAG: %[[#I8_Arg:]] = OpFunctionParameter %[[#I8]]
  ; CHECK-NATIVE-DAG: %[[#I16_Arg:]] = OpFunctionParameter %[[#I16]]
  ; CHECK-NATIVE-DAG: %[[#I32_Arg:]] = OpFunctionParameter %[[#I32]]
  ; CHECK-NATIVE-DAG: %[[#I64_Arg:]] = OpFunctionParameter %[[#I64]]

  ; CHECK-EMULATION-DAG: %[[#I8_Arg:]] = OpFunctionParameter %[[#I8]]
  ; CHECK-EMULATION-DAG: %[[#I16_Arg:]] = OpFunctionParameter %[[#I16]]
  ; CHECK-EMULATION-DAG: %[[#I32_Arg:]] = OpFunctionParameter %[[#I32]]
  ; CHECK-EMULATION-DAG: %[[#I64_Arg:]] = OpFunctionParameter %[[#I64]]

  ; CHECK-NATIVE: OpBitReverse  %[[#I8]] %[[#I8_Arg]]
  ; CHECK-EMULATION: OpShiftRightLogical %[[#I8]] %[[#I8_Arg]]
  ; CHECK-EMULATION: OpShiftLeftLogical %[[#I8:]]
  ; CHECK-EMULATION: OpBitwiseAnd %[[#I8:]]
  ; CHECK-EMULATION: OpBitwiseOr %[[#I8:]]

  %call8 = call i8 @llvm.bitreverse.i8(i8 %a8)
  store i8 %call8, ptr addrspace(1) %res, align 1

  ; CHECK-NATIVE: OpBitReverse  %[[#I16]] %[[#I16_Arg]]
  ; CHECK-EMULATION: OpShiftRightLogical %[[#I16]] %[[#I16_Arg]]
  ; CHECK-EMULATION: OpShiftLeftLogical %[[#I16:]]
  ; CHECK-EMULATION: OpBitwiseAnd %[[#I16:]]
  ; CHECK-EMULATION: OpBitwiseOr %[[#I16:]]

  %call16 = call i16 @llvm.bitreverse.i16(i16 %a16)
  store i16 %call16, ptr addrspace(1) %res, align 2

  ; CHECK-NATIVE: OpBitReverse  %[[#I32]] %[[#I32_Arg]]
  ; CHECK-EMULATION: OpShiftRightLogical %[[#I32]] %[[#I32_Arg]]
  ; CHECK-EMULATION: OpShiftLeftLogical %[[#I32:]]
  ; CHECK-EMULATION: OpBitwiseAnd %[[#I32:]]
  ; CHECK-EMULATION: OpBitwiseOr %[[#I32:]]

  %call32 = call i32 @llvm.bitreverse.i32(i32 %a32)
  store i32 %call32, ptr addrspace(1) %res, align 4

  ; CHECK-NATIVE: OpBitReverse  %[[#I64]] %[[#I64_Arg]]
  ; CHECK-EMULATION: OpShiftRightLogical %[[#I64]] %[[#I64_Arg]]
  ; CHECK-EMULATION: OpShiftLeftLogical %[[#I64:]]
  ; CHECK-EMULATION: OpBitwiseAnd %[[#I64:]]
  ; CHECK-EMULATION: OpBitwiseOr %[[#I64:]]

  %call64 = call i64 @llvm.bitreverse.i64(i64 %a64)
  store i64 %call64, ptr addrspace(1) %res, align 8

  ; CHECK-NATIVE-LABEL: OpFunctionEnd
  ; CHECK-EMULATION-LABEL: OpFunctionEnd
  ret void
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 2-element vectors
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

define spir_kernel void @test_vector_2(<2 x i8> %a8, <2 x i16> %a16, <2 x i32> %a32, <2 x i64> %a64, ptr addrspace(1) %res) #0 {
entry:
  ; CHECK-NATIVE-LABEL: Begin function test_vector_2
  ; CHECK-EMULATION-LABEL: Begin function test_vector_2
  ; CHECK-EMULATION-NOT: OpBitReverse

  ; CHECK-NATIVE-DAG: %[[#V2I8_Arg:]] = OpFunctionParameter %[[#V2I8]]
  ; CHECK-NATIVE-DAG: %[[#V2I16_Arg:]] = OpFunctionParameter %[[#V2I16]]
  ; CHECK-NATIVE-DAG: %[[#V2I32_Arg:]] = OpFunctionParameter %[[#V2I32]]
  ; CHECK-NATIVE-DAG: %[[#V2I64_Arg:]] = OpFunctionParameter %[[#V2I64]]

  ; CHECK-EMULATION-DAG: %[[#V2I8_Arg:]] = OpFunctionParameter %[[#V2I8]]
  ; CHECK-EMULATION-DAG: %[[#V2I16_Arg:]] = OpFunctionParameter %[[#V2I16]]
  ; CHECK-EMULATION-DAG: %[[#V2I32_Arg:]] = OpFunctionParameter %[[#V2I32]]
  ; CHECK-EMULATION-DAG: %[[#V2I64_Arg:]] = OpFunctionParameter %[[#V2I64]]


  ; CHECK-NATIVE: OpBitReverse  %[[#V2I8]] %[[#V2I8_Arg]]
  ; CHECK-EMULATION: OpShiftRightLogical %[[#V2I8]] %[[#V2I8_Arg]]
  ; CHECK-EMULATION: OpShiftLeftLogical %[[#V2I8:]]
  ; CHECK-EMULATION: OpBitwiseAnd %[[#V2I8:]]
  ; CHECK-EMULATION: OpBitwiseOr %[[#V2I8:]]

  %call8 = call <2 x i8> @llvm.bitreverse.v2i8(<2 x i8> %a8)
  store <2 x i8> %call8, ptr addrspace(1) %res, align 2
  
  ; CHECK-NATIVE: OpBitReverse  %[[#V2I16]] %[[#V2I16_Arg]]
  ; CHECK-EMULATION: OpShiftRightLogical %[[#V2I16]] %[[#V2I16_Arg]]
  ; CHECK-EMULATION: OpShiftLeftLogical %[[#V2I16:]]
  ; CHECK-EMULATION: OpBitwiseAnd %[[#V2I16:]]
  ; CHECK-EMULATION: OpBitwiseOr %[[#V2I16:]]
  %call16 = call <2 x i16> @llvm.bitreverse.v2i16(<2 x i16> %a16)
  store <2 x i16> %call16, ptr addrspace(1) %res, align 4

  ; CHECK-NATIVE: OpBitReverse  %[[#V2I32]] %[[#V2I32_Arg]]
  ; CHECK-EMULATION: OpShiftRightLogical %[[#V2I32]] %[[#V2I32_Arg]]
  ; CHECK-EMULATION: OpShiftLeftLogical %[[#V2I32:]]
  ; CHECK-EMULATION: OpBitwiseAnd %[[#V2I32:]]
  ; CHECK-EMULATION: OpBitwiseOr %[[#V2I32:]]
  %call32 = call <2 x i32> @llvm.bitreverse.v2i32(<2 x i32> %a32)
  store <2 x i32> %call32, ptr addrspace(1) %res, align 8

  ; CHECK-NATIVE: OpBitReverse  %[[#V2I64]] %[[#V2I64_Arg]]
  ; CHECK-EMULATION: OpShiftRightLogical %[[#V2I64]] %[[#V2I64_Arg]]
  ; CHECK-EMULATION: OpShiftLeftLogical %[[#V2I64:]]
  ; CHECK-EMULATION: OpBitwiseAnd %[[#V2I64:]]
  ; CHECK-EMULATION: OpBitwiseOr %[[#V2I64:]]
  %call64 = call <2 x i64> @llvm.bitreverse.v2i64(<2 x i64> %a64)
  store <2 x i64> %call64, ptr addrspace(1) %res, align 16

  ; CHECK-NATIVE-LABEL: OpFunctionEnd
  ; CHECK-EMULATION-LABEL: OpFunctionEnd
  ret void
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 4-element vectors
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

define spir_kernel void @test_vector_4(<4 x i8> %a8, <4 x i16> %a16, <4 x i32> %a32, <4 x i64> %a64, ptr addrspace(1) %res) #0 {
entry:
  ; CHECK-NATIVE-LABEL: Begin function test_vector_4
  ; CHECK-EMULATION-LABEL: Begin function test_vector_4
  ; CHECK-EMULATION-NOT: OpBitReverse

  ; CHECK-NATIVE-DAG: %[[#V4I8_Arg:]] = OpFunctionParameter %[[#V4I8]]
  ; CHECK-NATIVE-DAG: %[[#V4I16_Arg:]] = OpFunctionParameter %[[#V4I16]]
  ; CHECK-NATIVE-DAG: %[[#V4I32_Arg:]] = OpFunctionParameter %[[#V4I32]]
  ; CHECK-NATIVE-DAG: %[[#V4I64_Arg:]] = OpFunctionParameter %[[#V4I64]]

  ; CHECK-EMULATION-DAG: %[[#V4I8_Arg:]] = OpFunctionParameter %[[#V4I8]]
  ; CHECK-EMULATION-DAG: %[[#V4I16_Arg:]] = OpFunctionParameter %[[#V4I16]]
  ; CHECK-EMULATION-DAG: %[[#V4I32_Arg:]] = OpFunctionParameter %[[#V4I32]]
  ; CHECK-EMULATION-DAG: %[[#V4I64_Arg:]] = OpFunctionParameter %[[#V4I64]]


  ; CHECK-NATIVE: OpBitReverse  %[[#V4I8]] %[[#V4I8_Arg]]
  ; CHECK-EMULATION: OpShiftRightLogical %[[#V4I8]] %[[#V4I8_Arg]]
  ; CHECK-EMULATION: OpShiftLeftLogical %[[#V4I8:]]
  ; CHECK-EMULATION: OpBitwiseAnd %[[#V4I8:]]
  ; CHECK-EMULATION: OpBitwiseOr %[[#V4I8:]]

  %call8 = call <4 x i8> @llvm.bitreverse.v4i8(<4 x i8> %a8)
  store <4 x i8> %call8, ptr addrspace(1) %res, align 2
  
  ; CHECK-NATIVE: OpBitReverse  %[[#V4I16]] %[[#V4I16_Arg]]
  ; CHECK-EMULATION: OpShiftRightLogical %[[#V4I16]] %[[#V4I16_Arg]]
  ; CHECK-EMULATION: OpShiftLeftLogical %[[#V4I16:]]
  ; CHECK-EMULATION: OpBitwiseAnd %[[#V4I16:]]
  ; CHECK-EMULATION: OpBitwiseOr %[[#V4I16:]]
  %call16 = call <4 x i16> @llvm.bitreverse.v4i16(<4 x i16> %a16)
  store <4 x i16> %call16, ptr addrspace(1) %res, align 4

  ; CHECK-NATIVE: OpBitReverse  %[[#V4I32]] %[[#V4I32_Arg]]
  ; CHECK-EMULATION: OpShiftRightLogical %[[#V4I32]] %[[#V4I32_Arg]]
  ; CHECK-EMULATION: OpShiftLeftLogical %[[#V4I32:]]
  ; CHECK-EMULATION: OpBitwiseAnd %[[#V4I32:]]
  ; CHECK-EMULATION: OpBitwiseOr %[[#V4I32:]]
  %call32 = call <4 x i32> @llvm.bitreverse.v4i32(<4 x i32> %a32)
  store <4 x i32> %call32, ptr addrspace(1) %res, align 8

  ; CHECK-NATIVE: OpBitReverse  %[[#V4I64]] %[[#V4I64_Arg]]
  ; CHECK-EMULATION: OpShiftRightLogical %[[#V4I64]] %[[#V4I64_Arg]]
  ; CHECK-EMULATION: OpShiftLeftLogical %[[#V4I64:]]
  ; CHECK-EMULATION: OpBitwiseAnd %[[#V4I64:]]
  ; CHECK-EMULATION: OpBitwiseOr %[[#V4I64:]]
  %call64 = call <4 x i64> @llvm.bitreverse.v4i64(<4 x i64> %a64)
  store <4 x i64> %call64, ptr addrspace(1) %res, align 16

  ; CHECK-NATIVE-LABEL: OpFunctionEnd
  ; CHECK-EMULATION-LABEL: OpFunctionEnd
  ret void
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 8-element vectors
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

define spir_kernel void @test_vector_8(<8 x i8> %a8, <8 x i16> %a16, <8 x i32> %a32, <8 x i64> %a64, ptr addrspace(1) %res) #0 {
entry:
  ; CHECK-NATIVE-LABEL: Begin function test_vector_8
  ; CHECK-EMULATION-LABEL: Begin function test_vector_8
  ; CHECK-EMULATION-NOT: OpBitReverse

  ; CHECK-NATIVE-DAG: %[[#V8I8_Arg:]] = OpFunctionParameter %[[#V8I8]]
  ; CHECK-NATIVE-DAG: %[[#V8I16_Arg:]] = OpFunctionParameter %[[#V8I16]]
  ; CHECK-NATIVE-DAG: %[[#V8I32_Arg:]] = OpFunctionParameter %[[#V8I32]]
  ; CHECK-NATIVE-DAG: %[[#V8I64_Arg:]] = OpFunctionParameter %[[#V8I64]]

  ; CHECK-EMULATION-DAG: %[[#V8I8_Arg:]] = OpFunctionParameter %[[#V8I8]]
  ; CHECK-EMULATION-DAG: %[[#V8I16_Arg:]] = OpFunctionParameter %[[#V8I16]]
  ; CHECK-EMULATION-DAG: %[[#V8I32_Arg:]] = OpFunctionParameter %[[#V8I32]]
  ; CHECK-EMULATION-DAG: %[[#V8I64_Arg:]] = OpFunctionParameter %[[#V8I64]]


  ; CHECK-NATIVE: OpBitReverse  %[[#V8I8]] %[[#V8I8_Arg]]
  ; CHECK-EMULATION: OpShiftRightLogical %[[#V8I8]] %[[#V8I8_Arg]]
  ; CHECK-EMULATION: OpShiftLeftLogical %[[#V8I8:]]
  ; CHECK-EMULATION: OpBitwiseAnd %[[#V8I8:]]
  ; CHECK-EMULATION: OpBitwiseOr %[[#V8I8:]]

  %call8 = call <8 x i8> @llvm.bitreverse.v8i8(<8 x i8> %a8)
  store <8 x i8> %call8, ptr addrspace(1) %res, align 2
  
  ; CHECK-NATIVE: OpBitReverse  %[[#V8I16]] %[[#V8I16_Arg]]
  ; CHECK-EMULATION: OpShiftRightLogical %[[#V8I16]] %[[#V8I16_Arg]]
  ; CHECK-EMULATION: OpShiftLeftLogical %[[#V8I16:]]
  ; CHECK-EMULATION: OpBitwiseAnd %[[#V8I16:]]
  ; CHECK-EMULATION: OpBitwiseOr %[[#V8I16:]]
  %call16 = call <8 x i16> @llvm.bitreverse.v8i16(<8 x i16> %a16)
  store <8 x i16> %call16, ptr addrspace(1) %res, align 4

  ; CHECK-NATIVE: OpBitReverse  %[[#V8I32]] %[[#V8I32_Arg]]
  ; CHECK-EMULATION: OpShiftRightLogical %[[#V8I32]] %[[#V8I32_Arg]]
  ; CHECK-EMULATION: OpShiftLeftLogical %[[#V8I32:]]
  ; CHECK-EMULATION: OpBitwiseAnd %[[#V8I32:]]
  ; CHECK-EMULATION: OpBitwiseOr %[[#V8I32:]]
  %call32 = call <8 x i32> @llvm.bitreverse.v8i32(<8 x i32> %a32)
  store <8 x i32> %call32, ptr addrspace(1) %res, align 8

  ; CHECK-NATIVE: OpBitReverse  %[[#V8I64]] %[[#V8I64_Arg]]
  ; CHECK-EMULATION: OpShiftRightLogical %[[#V8I64]] %[[#V8I64_Arg]]
  ; CHECK-EMULATION: OpShiftLeftLogical %[[#V8I64:]]
  ; CHECK-EMULATION: OpBitwiseAnd %[[#V8I64:]]
  ; CHECK-EMULATION: OpBitwiseOr %[[#V8I64:]]
  %call64 = call <8 x i64> @llvm.bitreverse.v8i64(<8 x i64> %a64)
  store <8 x i64> %call64, ptr addrspace(1) %res, align 16

  ; CHECK-NATIVE-LABEL: OpFunctionEnd
  ; CHECK-EMULATION-LABEL: OpFunctionEnd
  ret void
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 16-element vectors
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

define spir_kernel void @test_vector_16(<16 x i8> %a8, <16 x i16> %a16, <16 x i32> %a32, <16 x i64> %a64, ptr addrspace(1) %res) #0 {
entry:
  ; CHECK-NATIVE-LABEL: Begin function test_vector_16
  ; CHECK-EMULATION-LABEL: Begin function test_vector_16
  ; CHECK-EMULATION-NOT: OpBitReverse

  ; CHECK-NATIVE-DAG: %[[#V16I8_Arg:]] = OpFunctionParameter %[[#V16I8]]
  ; CHECK-NATIVE-DAG: %[[#V16I16_Arg:]] = OpFunctionParameter %[[#V16I16]]
  ; CHECK-NATIVE-DAG: %[[#V16I32_Arg:]] = OpFunctionParameter %[[#V16I32]]
  ; CHECK-NATIVE-DAG: %[[#V16I64_Arg:]] = OpFunctionParameter %[[#V16I64]]

  ; CHECK-EMULATION-DAG: %[[#V16I8_Arg:]] = OpFunctionParameter %[[#V16I8]]
  ; CHECK-EMULATION-DAG: %[[#V16I16_Arg:]] = OpFunctionParameter %[[#V16I16]]
  ; CHECK-EMULATION-DAG: %[[#V16I32_Arg:]] = OpFunctionParameter %[[#V16I32]]
  ; CHECK-EMULATION-DAG: %[[#V16I64_Arg:]] = OpFunctionParameter %[[#V16I64]]


  ; CHECK-NATIVE: OpBitReverse  %[[#V16I8]] %[[#V16I8_Arg]]
  ; CHECK-EMULATION: OpShiftRightLogical %[[#V16I8]] %[[#V16I8_Arg]]
  ; CHECK-EMULATION: OpShiftLeftLogical %[[#V16I8:]]
  ; CHECK-EMULATION: OpBitwiseAnd %[[#V16I8:]]
  ; CHECK-EMULATION: OpBitwiseOr %[[#V16I8:]]

  %call8 = call <16 x i8> @llvm.bitreverse.v16i8(<16 x i8> %a8)
  store <16 x i8> %call8, ptr addrspace(1) %res, align 2
  
  ; CHECK-NATIVE: OpBitReverse  %[[#V16I16]] %[[#V16I16_Arg]]
  ; CHECK-EMULATION: OpShiftRightLogical %[[#V16I16]] %[[#V16I16_Arg]]
  ; CHECK-EMULATION: OpShiftLeftLogical %[[#V16I16:]]
  ; CHECK-EMULATION: OpBitwiseAnd %[[#V16I16:]]
  ; CHECK-EMULATION: OpBitwiseOr %[[#V16I16:]]
  %call16 = call <16 x i16> @llvm.bitreverse.v16i16(<16 x i16> %a16)
  store <16 x i16> %call16, ptr addrspace(1) %res, align 4

  ; CHECK-NATIVE: OpBitReverse  %[[#V16I32]] %[[#V16I32_Arg]]
  ; CHECK-EMULATION: OpShiftRightLogical %[[#V16I32]] %[[#V16I32_Arg]]
  ; CHECK-EMULATION: OpShiftLeftLogical %[[#V16I32:]]
  ; CHECK-EMULATION: OpBitwiseAnd %[[#V16I32:]]
  ; CHECK-EMULATION: OpBitwiseOr %[[#V16I32:]]
  %call32 = call <16 x i32> @llvm.bitreverse.v16i32(<16 x i32> %a32)
  store <16 x i32> %call32, ptr addrspace(1) %res, align 8

  ; CHECK-NATIVE: OpBitReverse  %[[#V16I64]] %[[#V16I64_Arg]]
  ; CHECK-EMULATION: OpShiftRightLogical %[[#V16I64]] %[[#V16I64_Arg]]
  ; CHECK-EMULATION: OpShiftLeftLogical %[[#V16I64:]]
  ; CHECK-EMULATION: OpBitwiseAnd %[[#V16I64:]]
  ; CHECK-EMULATION: OpBitwiseOr %[[#V16I64:]]
  %call64 = call <16 x i64> @llvm.bitreverse.v16i64(<16 x i64> %a64)
  store <16 x i64> %call64, ptr addrspace(1) %res, align 16

  ; CHECK-NATIVE-LABEL: OpFunctionEnd
  ; CHECK-EMULATION-LABEL: OpFunctionEnd
  ret void
}
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; Declarations
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; Scalar
declare i8 @llvm.bitreverse.i8(i8)
declare i16 @llvm.bitreverse.i16(i16)
declare i32 @llvm.bitreverse.i32(i32)
declare i64 @llvm.bitreverse.i64(i64)

; 2-element vectors
declare <2 x i8>  @llvm.bitreverse.v2i8(<2 x i8>)
declare <2 x i16> @llvm.bitreverse.v2i16(<2 x i16>)
declare <2 x i32> @llvm.bitreverse.v2i32(<2 x i32>)
declare <2 x i64> @llvm.bitreverse.v2i64(<2 x i64>)

; 4-element vector declarations
declare <4 x i8>  @llvm.bitreverse.v4i8(<4 x i8>)
declare <4 x i16> @llvm.bitreverse.v4i16(<4 x i16>)
declare <4 x i32> @llvm.bitreverse.v4i32(<4 x i32>)
declare <4 x i64> @llvm.bitreverse.v4i64(<4 x i64>)

; 8-element vector declarations
declare <8 x i8>  @llvm.bitreverse.v8i8(<8 x i8>)
declare <8 x i16> @llvm.bitreverse.v8i16(<8 x i16>)
declare <8 x i32> @llvm.bitreverse.v8i32(<8 x i32>)
declare <8 x i64> @llvm.bitreverse.v8i64(<8 x i64>)

; 16-element vector declarations
declare <16 x i8>  @llvm.bitreverse.v16i8(<16 x i8>)
declare <16 x i16> @llvm.bitreverse.v16i16(<16 x i16>)
declare <16 x i32> @llvm.bitreverse.v16i32(<16 x i32>)
declare <16 x i64> @llvm.bitreverse.v16i64(<16 x i64>)
