; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#F32:]] = OpTypeFloat 32

; CHECK: %[[#ARG:]] = OpFunctionParameter %[[#F32]]
; CHECK-NOT: OpBitcast
; CHECK: OpReturnValue %[[#ARG]]
define float @identity_roundtrip(float %x) {
  %a = bitcast float %x to i32
  %b = bitcast i32 %a to float
  ret float %b
}

; CHECK: %[[#ARG2:]] = OpFunctionParameter %[[#I32]]
; CHECK: %[[#BC:]] = OpBitcast %[[#F32]] %[[#ARG2]]
; CHECK-NOT: OpBitcast
; CHECK: OpReturnValue %[[#BC]]
define float @odd_chain(i32 %x) {
  %a = bitcast i32 %x to float
  %b = bitcast float %a to i32
  %c = bitcast i32 %b to float
  ret float %c
}
