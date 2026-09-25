; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_poison_freeze %s -o - | FileCheck %s
; RUN: llc -O0 -mtriple=spirv-vulkan-unknown --spirv-ext=+SPV_KHR_poison_freeze %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_poison_freeze %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpCapability PoisonFreezeKHR
; CHECK-DAG: OpExtension "SPV_KHR_poison_freeze"

; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#ARR:]] = OpTypeArray %[[#I32]] %[[#]]

; CHECK-DAG: %[[#INNER:]] = OpTypeArray %[[#I32]] %[[#]]
; CHECK-DAG: %[[#NEST:]] = OpTypeArray %[[#INNER]] %[[#]]

; CHECK-DAG: %[[#PI:]] = OpPoisonKHR %[[#I32]]
; CHECK-DAG: %[[#C1:]] = OpConstant %[[#I32]] 1
; CHECK-DAG: %[[#C3:]] = OpConstant %[[#I32]] 3
; CHECK-DAG: %[[#C4:]] = OpConstant %[[#I32]] 4
; CHECK-DAG: %[[#MIX:]] = OpConstantComposite %[[#ARR]] %[[#C1]] %[[#PI]] %[[#C3]] %[[#C4]]

; CHECK-DAG: %[[#PINNER:]] = OpPoisonKHR %[[#INNER]]
; CHECK-DAG: %[[#PARTIAL:]] = OpConstantComposite %[[#INNER]] %[[#C1]] %[[#PI]]
; CHECK-DAG: %[[#NESTMIX:]] = OpConstantComposite %[[#NEST]] %[[#PARTIAL]] %[[#PINNER]]

; CHECK: OpFunction
; CHECK: OpStore %[[#]] %[[#MIX]]

; CHECK: OpFunction
; CHECK: OpStore %[[#]] %[[#NESTMIX]]

%arr = type [4 x i32]
%nest = type [2 x [2 x i32]]

@g_partial = global %arr [i32 1, i32 poison, i32 3, i32 4]

define void @poison_partial_composite(ptr %dst) {
  store %arr [i32 1, i32 poison, i32 3, i32 4], ptr %dst
  ret void
}

define void @poison_nested_composite(ptr %dst) {
  store %nest [[2 x i32] [i32 1, i32 poison], [2 x i32] poison], ptr %dst
  ret void
}
