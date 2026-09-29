; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-vulkan-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-vulkan-unknown %s -o - -filetype=obj | spirv-val %}

; Check that a poison element nested inside a constant aggregate is lowered to
; an OpUndef placeholder, instead of reaching IRTranslator as an untranslatable
; spv.const.composite operand and crashing.

; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#ARR:]] = OpTypeArray %[[#I32]] %[[#]]

; CHECK-DAG: %[[#INNER:]] = OpTypeArray %[[#I32]] %[[#]]
; CHECK-DAG: %[[#NEST:]] = OpTypeArray %[[#INNER]] %[[#]]

; CHECK-DAG: %[[#UI:]] = OpUndef %[[#I32]]
; CHECK-DAG: %[[#C1:]] = OpConstant %[[#I32]] 1
; CHECK-DAG: %[[#C3:]] = OpConstant %[[#I32]] 3
; CHECK-DAG: %[[#C4:]] = OpConstant %[[#I32]] 4
; CHECK-DAG: %[[#MIX:]] = OpConstantComposite %[[#ARR]] %[[#C1]] %[[#UI]] %[[#C3]] %[[#C4]]

; CHECK-DAG: %[[#UINNER:]] = OpUndef %[[#INNER]]
; CHECK-DAG: %[[#COMPOSITE:]] = OpConstantComposite %[[#INNER]] %[[#C1]] %[[#UI]]
; CHECK-DAG: %[[#NESTMIX:]] = OpConstantComposite %[[#NEST]] %[[#COMPOSITE]] %[[#UINNER]]

; CHECK: OpFunction
; CHECK: OpStore %[[#]] %[[#MIX]]

; CHECK: OpFunction
; CHECK: OpStore %[[#]] %[[#NESTMIX]]

%arr = type [4 x i32]
%nest = type [2 x [2 x i32]]

@g = global %arr [i32 1, i32 poison, i32 3, i32 4]

define void @undef_composite(ptr %dst) {
  store %arr [i32 1, i32 poison, i32 3, i32 4], ptr %dst
  ret void
}

define void @undef_nested_composite(ptr %dst) {
  store %nest [[2 x i32] [i32 1, i32 poison], [2 x i32] poison], ptr %dst
  ret void
}
