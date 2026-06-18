; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Verify that <1 x T> types are scalarized to T since SPIR-V doesn't support
; single-element vectors.

; CHECK-DAG: %[[#INT16:]] = OpTypeInt 16 0
; CHECK-DAG: %[[#FLOAT64:]] = OpTypeFloat 64
; CHECK-DAG: %[[#VEC4:]] = OpTypeVector %[[#FLOAT64]] 4
; CHECK-DAG: %[[#FNTY:]] = OpTypeFunction %[[#VEC4]] %[[#INT16]]
; CHECK-DAG: %[[#ZERO:]] = OpConstantNull %[[#INT16]]

; CHECK: OpFunctionCall %[[#VEC4]] %[[#]] %[[#ZERO]]
define spir_func <4 x double> @caller() {
entry:
  %C = call <4 x double> @callee(<1 x i16> zeroinitializer)
  ret <4 x double> %C
}
declare <4 x double> @callee(<1 x i16>)

; CHECK: %[[#EXTRACT_RES:]] = OpFunctionParameter %[[#INT16]]
; CHECK: OpReturnValue %[[#EXTRACT_RES]]
define spir_func i16 @test_extractelement(<1 x i16> %v) {
entry:
  %e = extractelement <1 x i16> %v, i32 0
  ret i16 %e
}

; CHECK: %[[#INSERT_VAL:]] = OpFunctionParameter %[[#INT16]]
; CHECK: OpReturnValue %[[#INSERT_VAL]]
define spir_func <1 x i16> @test_insertelement(i16 %val) {
entry:
  %v = insertelement <1 x i16> poison, i16 %val, i32 0
  ret <1 x i16> %v
}

; CHECK: %[[#SHUF_PARAM:]] = OpFunctionParameter %[[#INT16]]
; CHECK: OpReturnValue %[[#SHUF_PARAM]]
define spir_func <1 x i16> @test_shufflevector(<1 x i16> %v) {
entry:
  %s = shufflevector <1 x i16> %v, <1 x i16> poison, <1 x i32> zeroinitializer
  ret <1 x i16> %s
}

; CHECK: %[[#LHS_PARAM:]] = OpFunctionParameter %[[#INT16]]
; CHECK: %[[#RHS_PARAM:]] = OpFunctionParameter %[[#INT16]]
; CHECK: %[[#RET:]] = OpIAdd %[[#INT16]] %[[#LHS_PARAM]] %[[#RHS_PARAM]]
; CHECK: OpReturnValue %[[#RET]]
define spir_func <1 x i16> @test_arithm(<1 x i16> %v1, <1 x i16> %v2) {
entry:
  %s = add <1 x i16> %v1, %v2
  ret <1 x i16> %s
}
