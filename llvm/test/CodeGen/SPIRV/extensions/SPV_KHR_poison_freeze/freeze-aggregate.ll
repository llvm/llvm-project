; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_poison_freeze %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_poison_freeze %s -o - -filetype=obj | spirv-val %}

; An aggregate freeze takes its result type from its operand and is lowered like
; aggregate PHIs and selects. With SPV_KHR_poison_freeze it becomes an
; OpFreezeKHR over the composite type.

; CHECK-DAG: OpCapability PoisonFreezeKHR
; CHECK-DAG: OpExtension "SPV_KHR_poison_freeze"

; CHECK-DAG: %[[#Float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#Int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Two:]] = OpConstant %[[#Int]] 2
; CHECK-DAG: %[[#Array:]] = OpTypeArray %[[#Float]] %[[#Two]]
; CHECK-DAG: %[[#Struct:]] = OpTypeStruct %[[#Float]] %[[#Float]]

; CHECK: %[[#L:]] = OpLoad %[[#Array]]
; CHECK: %[[#FL:]] = OpFreezeKHR %[[#Array]] %[[#L]]
; CHECK: OpCompositeExtract %[[#Float]] %[[#FL]] 0
define spir_kernel void @freeze_loaded(ptr addrspace(1) %out, ptr addrspace(1) %pa) {
  %a = load [2 x float], ptr addrspace(1) %pa
  %v = freeze [2 x float] %a
  %e0 = extractvalue [2 x float] %v, 0
  store float %e0, ptr addrspace(1) %out
  ret void
}

; CHECK: %[[#Ins:]] = OpCompositeInsert %[[#Struct]] %[[#]] %[[#]] 1
; CHECK: %[[#FI:]] = OpFreezeKHR %[[#Struct]] %[[#Ins]]
; CHECK: OpCompositeExtract %[[#Float]] %[[#FI]] 1
define spir_kernel void @freeze_insertvalue(ptr addrspace(1) %out, float %x) {
  %ins = insertvalue { float, float } zeroinitializer, float %x, 1
  %v = freeze { float, float } %ins
  %e1 = extractvalue { float, float } %v, 1
  store float %e1, ptr addrspace(1) %out
  ret void
}

; CHECK: %[[#S:]] = OpLoad %[[#Array]]
; CHECK: %[[#FS:]] = OpFreezeKHR %[[#Array]] %[[#S]]
; CHECK: OpStore %[[#]] %[[#FS]]
define spir_kernel void @freeze_store_direct(ptr addrspace(1) %out, ptr addrspace(1) %pa) {
  %a = load [2 x float], ptr addrspace(1) %pa
  %v = freeze [2 x float] %a
  store [2 x float] %v, ptr addrspace(1) %out
  ret void
}
