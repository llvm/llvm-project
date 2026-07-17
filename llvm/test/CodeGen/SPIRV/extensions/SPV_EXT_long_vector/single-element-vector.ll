; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_long_vector %s -o - | FileCheck %s
; spirv-val seems to have problems reading OpTypeVectorIdEXT correctly, enable once fixed
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_long_vector %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#INT16:]] = OpTypeInt 16 0
; CHECK-DAG: %[[#FLOAT64:]] = OpTypeFloat 64
; CHECK-DAG: %[[#VEC4:]] = OpTypeVector %[[#FLOAT64]] 4
; CHECK-DAG: %[[#VEC1:]] = OpTypeVectorIdEXT %[[#INT16]] 1
; CHECK-DAG: %[[#FNTY:]] = OpTypeFunction %[[#VEC4]] %[[#VEC1]]
; CHECK-DAG: %[[#ZERO:]] = OpConstantNull %[[#VEC1]]

; CHECK: OpFunctionCall %[[#VEC4]] %[[#]] %[[#ZERO]]
define spir_func <4 x double> @caller() {
entry:
  %C = call <4 x double> @callee(<1 x i16> zeroinitializer)
  ret <4 x double> %C
}
declare <4 x double> @callee(<1 x i16>)

; CHECK: %[[#V:]] = OpFunctionParameter %[[#VEC1]]
; CHECK: %[[#EXTRACT_RES:]] = OpCompositeExtract %[[#INT16]] %[[#V]] 0
; CHECK: OpReturnValue %[[#EXTRACT_RES]]
define spir_func i16 @test_extractelement(<1 x i16> %v) {
entry:
  %e = extractelement <1 x i16> %v, i32 0
  ret i16 %e
}

; CHECK: %[[#VAL:]] = OpFunctionParameter %[[#INT16]]
; CHECK: %[[#INSERT_VAL:]] = OpCompositeInsert %[[#VEC1]] %[[#VAL]] %[[#]] 0
; CHECK: OpReturnValue %[[#INSERT_VAL]]
define spir_func <1 x i16> @test_insertelement(i16 %val) {
entry:
  %v = insertelement <1 x i16> poison, i16 %val, i32 0
  ret <1 x i16> %v
}

; CHECK: %[[#SHUF_PARAM:]] = OpFunctionParameter %[[#VEC1]]
; CHECK: OpReturnValue %[[#SHUF_PARAM]]
define spir_func <1 x i16> @test_shufflevector(<1 x i16> %v) {
entry:
  %s = shufflevector <1 x i16> %v, <1 x i16> poison, <1 x i32> zeroinitializer
  ret <1 x i16> %s
}

; CHECK: %[[#LHS_PARAM:]] = OpFunctionParameter %[[#VEC1]]
; CHECK: %[[#RHS_PARAM:]] = OpFunctionParameter %[[#VEC1]]
; CHECK: %[[#RET:]] = OpIAdd %[[#VEC1]] %[[#LHS_PARAM]] %[[#RHS_PARAM]]
; CHECK: OpReturnValue %[[#RET]]
define spir_func <1 x i16> @test_arithm(<1 x i16> %v1, <1 x i16> %v2) {
entry:
  %s = add <1 x i16> %v1, %v2
  ret <1 x i16> %s
}
