; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val %}

; Code here is an excerpt of clang/test/CodeGenHLSL/builtins/AddUint64.hlsl compiled for spirv using the following command
; clang -cc1 -finclude-default-header -triple spirv-unknown-vulkan-compute clang/test/CodeGenHLSL/builtins/AddUint64.hlsl -emit-llvm -disable-llvm-passes -o llvm/test/CodeGen/SPIRV/hlsl-intrinsics/uadd_with_overflow.ll

; CHECK-DAG: %[[#int_32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#vec2_int_32:]] = OpTypeVector %[[#int_32]] 2
; CHECK-DAG: %[[#bool:]] = OpTypeBool
; CHECK-DAG: %[[#const_i32_1:]] = OpConstant %[[#int_32]] 1{{$}}
; CHECK-DAG: %[[#struct_i32_i32:]] = OpTypeStruct %[[#int_32]] %[[#int_32]]
; CHECK-DAG: %[[#func_v2i32_v2i32_v2i32:]] = OpTypeFunction %[[#vec2_int_32]] %[[#vec2_int_32]] %[[#vec2_int_32]]
; CHECK-DAG: %[[#const_i32_0:]] = OpConstant %[[#int_32]] 0{{$}}
; CHECK-DAG: %[[#undef_v2i32:]] = OpUndef %[[#vec2_int_32]]
; CHECK-DAG: %[[#vec4_int_32:]] = OpTypeVector %[[#int_32]] 4
; CHECK-DAG: %[[#vec2_bool:]] = OpTypeVector %[[#bool]] 2
; CHECK-DAG: %[[#const_v2i32_0_0:]] = OpConstantComposite %[[#vec2_int_32]] %[[#const_i32_0]] %[[#const_i32_0]]
; CHECK-DAG: %[[#const_v2i32_1_1:]] = OpConstantComposite %[[#vec2_int_32]] %[[#const_i32_1]] %[[#const_i32_1]]
; CHECK-DAG: %[[#struct_v2i32_v2i32:]] = OpTypeStruct %[[#vec2_int_32]] %[[#vec2_int_32]]
; CHECK-DAG: %[[#func_v4i32_v4i32_v4i32:]] = OpTypeFunction %[[#vec4_int_32]] %[[#vec4_int_32]] %[[#vec4_int_32]]
; CHECK-DAG: %[[#undef_v4i32:]] = OpUndef %[[#vec4_int_32]]


define spir_func <2 x i32> @test_AddUint64_uint2(<2 x i32> %a, <2 x i32> %b) {
entry:
; CHECK: %[[#a:]] = OpFunctionParameter %[[#vec2_int_32]]
; CHECK: %[[#b:]] = OpFunctionParameter %[[#vec2_int_32]]
; CHECK: %[[#a_low:]] = OpCompositeExtract %[[#int_32]] %[[#a]] 0
; CHECK: %[[#a_high:]] = OpCompositeExtract %[[#int_32]] %[[#a]] 1
; CHECK: %[[#b_low:]] = OpCompositeExtract %[[#int_32]] %[[#b]] 0
; CHECK: %[[#b_high:]] = OpCompositeExtract %[[#int_32]] %[[#b]] 1
; CHECK: %[[#iaddcarry:]] = OpIAddCarry %[[#struct_i32_i32]] %[[#a_low]] %[[#b_low]]
; CHECK: %[[#lowsum:]] = OpCompositeExtract %[[#int_32]] %[[#iaddcarry]] 0
; CHECK: %[[#carry:]] = OpCompositeExtract %[[#int_32]] %[[#iaddcarry]] 1
; CHECK: %[[#carry_ne0:]] = OpINotEqual %[[#bool]] %[[#carry]] %[[#const_i32_0]]
; CHECK: %[[#select_1_or_0:]] = OpSelect %[[#int_32]] %[[#carry_ne0]] %[[#const_i32_1]] %[[#const_i32_0]]
; CHECK: %[[#highsum:]] = OpIAdd %[[#int_32]] %[[#a_high]] %[[#b_high]]
; CHECK: %[[#highsumpluscarry:]] = OpIAdd %[[#int_32]] %[[#highsum]] %[[#select_1_or_0]]
; CHECK: %[[#adduint64_upto0:]] = OpCompositeInsert %[[#vec2_int_32]] %[[#lowsum]] %[[#undef_v2i32]] 0
; CHECK: %[[#adduint64:]] = OpCompositeInsert %[[#vec2_int_32]] %[[#highsumpluscarry]] %[[#adduint64_upto0]] 1
; CHECK: OpReturnValue %[[#adduint64]]
;
  %LowA = extractelement <2 x i32> %a, i64 0
  %HighA = extractelement <2 x i32> %a, i64 1
  %LowB = extractelement <2 x i32> %b, i64 0
  %HighB = extractelement <2 x i32> %b, i64 1
  %3 = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %LowA, i32 %LowB)
  %4 = extractvalue { i32, i1 } %3, 1
  %5 = extractvalue { i32, i1 } %3, 0
  %CarryZExt = zext i1 %4 to i32
  %HighSum = add i32 %HighA, %HighB
  %HighSumPlusCarry = add i32 %HighSum, %CarryZExt
  %hlsl.AddUint64.upto0 = insertelement <2 x i32> poison, i32 %5, i64 0
  %hlsl.AddUint64 = insertelement <2 x i32> %hlsl.AddUint64.upto0, i32 %HighSumPlusCarry, i64 1
  ret <2 x i32> %hlsl.AddUint64
}

define spir_func <4 x i32> @test_AddUint64_uint4(<4 x i32> %a, <4 x i32> %b) #0 {
entry:
; CHECK: %[[#a:]] = OpFunctionParameter %[[#vec4_int_32]]
; CHECK: %[[#b:]] = OpFunctionParameter %[[#vec4_int_32]]
; CHECK: %[[#a_low:]] = OpVectorShuffle %[[#vec2_int_32]] %[[#a]] %[[#undef_v4i32]] 0 2
; CHECK: %[[#a_high:]] = OpVectorShuffle %[[#vec2_int_32]] %[[#a]] %[[#undef_v4i32]] 1 3
; CHECK: %[[#b_low:]] = OpVectorShuffle %[[#vec2_int_32]] %[[#b]] %[[#undef_v4i32]] 0 2
; CHECK: %[[#b_high:]] = OpVectorShuffle %[[#vec2_int_32]] %[[#b]] %[[#undef_v4i32]] 1 3
; CHECK: %[[#iaddcarry:]] = OpIAddCarry %[[#struct_v2i32_v2i32]] %[[#a_low]] %[[#vec2_int_32]]
; CHECK: %[[#lowsum:]] = OpCompositeExtract %[[#vec2_int_32]] %[[#iaddcarry]] 0
; CHECK: %[[#carry:]] = OpCompositeExtract %[[#vec2_int_32]] %[[#iaddcarry]] 1
; CHECK: %[[#carry_ne0:]] = OpINotEqual %[[#vec2_bool]] %[[#carry]] %[[#const_v2i32_0_0]]
; CHECK: %[[#select_1_or_0:]] = OpSelect %[[#vec2_int_32]] %[[#carry_ne0]] %[[#const_v2i32_1_1]] %[[#const_v2i32_0_0]]
; CHECK: %[[#highsum:]] = OpIAdd %[[#vec2_int_32]] %[[#a_high]] %[[#b_high]]
; CHECK: %[[#highsumpluscarry:]] = OpIAdd %[[#vec2_int_32]] %[[#highsum]] %[[#select_1_or_0]]
; CHECK: %[[#adduint64:]] = OpVectorShuffle %[[#vec4_int_32]] %[[#lowsum]] %[[#highsumpluscarry]] 0 2 1 3
; CHECK: OpReturnValue %[[#adduint64]]
;
  %LowA = shufflevector <4 x i32> %a, <4 x i32> poison, <2 x i32> <i32 0, i32 2>
  %HighA = shufflevector <4 x i32> %a, <4 x i32> poison, <2 x i32> <i32 1, i32 3>
  %LowB = shufflevector <4 x i32> %b, <4 x i32> poison, <2 x i32> <i32 0, i32 2>
  %HighB = shufflevector <4 x i32> %b, <4 x i32> poison, <2 x i32> <i32 1, i32 3>
  %3 = call { <2 x i32>, <2 x i1> } @llvm.uadd.with.overflow.v2i32(<2 x i32> %LowA, <2 x i32> %LowB)
  %4 = extractvalue { <2 x i32>, <2 x i1> } %3, 1
  %5 = extractvalue { <2 x i32>, <2 x i1> } %3, 0
  %CarryZExt = zext <2 x i1> %4 to <2 x i32>
  %HighSum = add <2 x i32> %HighA, %HighB
  %HighSumPlusCarry = add <2 x i32> %HighSum, %CarryZExt
  %hlsl.AddUint64 = shufflevector <2 x i32> %5, <2 x i32> %HighSumPlusCarry, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  ret <4 x i32> %hlsl.AddUint64
}
