; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_poison_freeze %s -o - | FileCheck %s --implicit-check-not=OpPoisonKHR
; RUN: llc -O0 -mtriple=spirv-vulkan-unknown --spirv-ext=+SPV_KHR_poison_freeze %s -o - | FileCheck %s --implicit-check-not=OpPoisonKHR
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_poison_freeze %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-vulkan-unknown --spirv-ext=+SPV_KHR_poison_freeze %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpCapability PoisonFreezeKHR
; CHECK-DAG: OpExtension "SPV_KHR_poison_freeze"

; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#V4:]] = OpTypeVector %[[#I32]] 4
; CHECK-DAG: %[[#ARR:]] = OpTypeArray %[[#I32]] %[[#]]

; CHECK-DAG: %[[#PI:]] = OpPoisonKHR %[[#I32]]
; CHECK-DAG: %[[#PV:]] = OpPoisonKHR %[[#V4]]
; CHECK-DAG: %[[#CC:]] = OpConstantComposite %[[#ARR]] %[[#PI]] %[[#PI]] %[[#PI]] %[[#PI]]

%arr = type [4 x i32]

define void @poison_scalar(ptr %dst) {
  store i32 poison, ptr %dst
  store i32 poison, ptr %dst
  ret void
}

define void @poison_vector(ptr %dst) {
  store <4 x i32> poison, ptr %dst
  store <4 x i32> poison, ptr %dst
  ret void
}

define void @poison_aggregate(ptr %dst) {
  store %arr poison, ptr %dst
  ret void
}
