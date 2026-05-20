; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_poison_freeze %s -o - | FileCheck %s
; RUN: llc -O0 -mtriple=spirv-vulkan-unknown --spirv-ext=+SPV_KHR_poison_freeze %s -o - | FileCheck %s

; spirv-val is intentionally not run: SPV_KHR_poison_freeze does not yet permit
; OpPoisonKHR as an OpConstantComposite constituent. Pending Khronos spec PR.

; CHECK-DAG: OpCapability PoisonFreezeKHR
; CHECK-DAG: OpExtension "SPV_KHR_poison_freeze"

; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#ARR:]] = OpTypeArray %[[#I32]] %[[#]]

; CHECK-DAG: %[[#PI:]] = OpPoisonKHR %[[#I32]]
; CHECK-DAG: %[[#C1:]] = OpConstant %[[#I32]] 1
; CHECK-DAG: %[[#C3:]] = OpConstant %[[#I32]] 3
; CHECK-DAG: %[[#C4:]] = OpConstant %[[#I32]] 4
; CHECK-DAG: %[[#MIX:]] = OpConstantComposite %[[#ARR]] %[[#C1]] %[[#PI]] %[[#C3]] %[[#C4]]

; CHECK: OpFunction
; CHECK: OpStore %[[#]] %[[#MIX]]

%arr = type [4 x i32]

@g_partial = global %arr [i32 1, i32 poison, i32 3, i32 4]

define void @poison_partial_composite(ptr %dst) {
  store %arr [i32 1, i32 poison, i32 3, i32 4], ptr %dst
  ret void
}
