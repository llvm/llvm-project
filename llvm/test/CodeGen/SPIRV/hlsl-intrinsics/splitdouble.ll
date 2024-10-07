; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Make sure lowering is correctly generating spirv code.

; CHECK-DAG: %[[#double:]] = OpTypeFloat 64
; CHECK-DAG: %[[#int_32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#scalar_function:]] = OpTypeFunction %[[#int_32]] %[[#double]]
; CHECK-DAG: %[[#vec_2_int_32:]] = OpTypeVector %[[#int_32]] 2
; CHECK-DAG: %[[#vec_4_int_32:]] = OpTypeVector %[[#int_32]] 4
; CHECK-DAG: %[[#vec_3_int_32:]] = OpTypeVector %[[#int_32]] 3
; CHECK-DAG: %[[#vec_3_double:]] = OpTypeVector %[[#double]] 3
; CHECK-DAG: %[[#vector_function:]] = OpTypeFunction %[[#vec_3_int_32]] %[[#vec_3_double]]
; CHECK-DAG: %[[#vec_2_double:]] = OpTypeVector %[[#double]] 2


define spir_func noundef i32 @test_scalar(double noundef %D) local_unnamed_addr {
entry:
  ; CHECK-LABEL: ; -- Begin function test_scalar
  ; CHECK: %[[#param:]] = OpFunctionParameter %[[#double]]
  ; CHECK: %[[#bitcast:]] = OpBitcast %[[#vec_2_int_32]] %[[#param]]
  %0 = bitcast double %D to <2 x i32>
  ; CHECK: %[[#]] = OpCompositeExtract %[[#int_32]] %[[#bitcast]] 0
  %1 = extractelement <2 x i32> %0, i64 0
  ; CHECK: %[[#]] = OpCompositeExtract %[[#int_32]] %[[#bitcast]] 1
  %2 = extractelement <2 x i32> %0, i64 1
  %add = add i32 %1, %2
  ret i32 %add
}


define spir_func noundef <3 x i32> @test_vector(<3 x double> noundef %D) local_unnamed_addr {
entry:
  ; CHECK-LABEL: ; -- Begin function test_vector
  ; CHECK: %[[#param:]] = OpFunctionParameter %[[#vec_3_double]]
  ; %[[#SHUFF1:]] = OpVectorShuffle %[[#vec_2_double]] %[[#param]] %[[#]] 0 1
  ; %[[#CAST1:]] = OpBitcast %[[#vec_4_int_32]] %[[#SHUFF1]]
  ; %[[#SHUFF2:]] = OpVectorShuffle %[[#vec_2_int_32]] %[[#CAST1]] %[[#]] 0 2
  ; %[[#SHUFF3:]] = OpVectorShuffle %[[#vec_2_int_32]] %[[#CAST1]] %[[#]] 1 3
  ; %[[#EXTRACT:]] = OpCompositeExtract %[[#double]] %[[#param]] 2
  ; %[[#CAST2:]] = OpBitcast %[[#vec_2_int_32]] %[[#EXTRACT]]
  ; %[[#]] = OpVectorShuffle %7 %[[#SHUFF2]] %[[#CAST2]] 0 1 2
  ; %[[#]] = OpVectorShuffle %7 %[[#SHUFF3]] %[[#CAST2]] 0 1 3
  %0 = shufflevector <3 x double> %D, <3 x double> poison, <2 x i32> <i32 0, i32 1>
  %1 = bitcast <2 x double> %0 to <4 x i32>
  %2 = shufflevector <4 x i32> %1, <4 x i32> poison, <2 x i32> <i32 0, i32 2>
  %3 = shufflevector <4 x i32> %1, <4 x i32> poison, <2 x i32> <i32 1, i32 3>
  %4 = extractelement <3 x double> %D, i64 2
  %5 = bitcast double %4 to <2 x i32>
  %6 = shufflevector <2 x i32> %2, <2 x i32> %5, <3 x i32> <i32 0, i32 1, i32 2>
  %7 = shufflevector <2 x i32> %3, <2 x i32> %5, <3 x i32> <i32 0, i32 1, i32 3>
  %add = add <3 x i32> %6, %7
  ret <3 x i32> %add
}
