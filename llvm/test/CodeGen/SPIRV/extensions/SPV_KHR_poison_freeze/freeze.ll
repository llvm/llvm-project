; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_poison_freeze %s -o - | FileCheck %s
; RUN: llc -O0 -mtriple=spirv-vulkan-unknown --spirv-ext=+SPV_KHR_poison_freeze %s -o - | FileCheck %s --check-prefix=VK --implicit-check-not=PoisonFreezeKHR --implicit-check-not=OpPoisonKHR --implicit-check-not=OpFreezeKHR
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_poison_freeze %s -o - -filetype=obj | spirv-val %}

; Vulkan does not allow the extension, so a request for it is ignored.
; VK: OpFunction

; CHECK-DAG: OpCapability PoisonFreezeKHR
; CHECK-DAG: OpExtension "SPV_KHR_poison_freeze"

; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#V4:]] = OpTypeVector %[[#I32]] 4

; CHECK-DAG: %[[#PI:]] = OpPoisonKHR %[[#I32]]
; CHECK-DAG: %[[#PV:]] = OpPoisonKHR %[[#V4]]

; CHECK: %[[#FS:]] = OpFreezeKHR %[[#I32]] %[[#PI]]
; CHECK: OpStore %[[#]] %[[#FS]]

; CHECK: %[[#FVP:]] = OpFreezeKHR %[[#V4]] %[[#PV]]
; CHECK: OpStore %[[#]] %[[#FVP]]

; CHECK: %[[#FV:]] = OpFreezeKHR %[[#V4]] %[[#]]
; CHECK: OpStore %[[#]] %[[#FV]]

define void @freeze_scalar_poison(ptr %dst) {
  %f = freeze i32 poison
  store i32 %f, ptr %dst
  ret void
}

define void @freeze_vector_poison(ptr %dst) {
  %f = freeze <4 x i32> poison
  store <4 x i32> %f, ptr %dst
  ret void
}

define void @freeze_vector(ptr %dst, <4 x i32> %v) {
  %f = freeze <4 x i32> %v
  store <4 x i32> %f, ptr %dst
  ret void
}
