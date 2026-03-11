; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_int4 %s -o - | FileCheck %s
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknow --spirv-ext=+SPV_INTEL_int4 %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpCapability Int4TypeINTEL
; CHECK-DAG: OpExtension "SPV_INTEL_int4"

; CHECK-DAG: %[[#i4:]] = OpTypeInt 4 0
; CHECK-DAG: %[[#bool:]] = OpTypeBool
; CHECK-DAG: %[[#scalar_func:]] = OpTypeFunction %[[#bool]] %[[#i4]] %[[#i4]]
; CHECK-DAG: %[[#vec_i4:]] = OpTypeVector %[[#i4]] 4
; CHECK-DAG: %[[#vec_bool:]] = OpTypeVector %[[#bool]] 4
; CHECK-DAG: %[[#vector_func:]] = OpTypeFunction %[[#vec_bool]] %[[#vec_i4]] %[[#vec_i4]]

; CHECK: %[[#test_scalar:]] = OpFunction %[[#bool]] None %[[#scalar_func]]
; CHECK: %[[#scalar_a:]] = OpFunctionParameter %[[#i4]]
; CHECK: %[[#scalar_b:]] = OpFunctionParameter %[[#i4]]
define spir_func i1 @test_scalar(i4 %a, i4 %b) {
entry:
; CHECK: %[[#scalar_entry:]] = OpLabel
  %res1 = icmp eq i4 %a, %b
; CHECK: %[[#scalar_cmp:]] = OpIEqual %[[#bool]] %[[#scalar_a]] %[[#scalar_b]]
  ret i1 %res1
}

; CHECK: %[[#test_vector:]] = OpFunction %[[#vec_bool]] None %[[#vector_func]]
; CHECK: %[[#vector_a:]] = OpFunctionParameter %[[#vec_i4]]
; CHECK: %[[#vector_b:]] = OpFunctionParameter %[[#vec_i4]]
define spir_func <4 x i1> @test_vector(<4 x i4> %a, <4 x i4> %b) {
entry:
; CHECK: %[[#vector_entry:]] = OpLabel
  %res2 = icmp eq <4 x i4> %a, %b
; CHECK: %[[#vector_cmp:]] = OpIEqual %[[#vec_bool]] %[[#vector_a]] %[[#vector_b]]
  ret <4 x i1> %res2
}
