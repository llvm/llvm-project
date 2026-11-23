; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; An aggregate freeze takes its result type from its operand and must be lowered
; like aggregate PHIs and selects.

; CHECK-DAG: %[[#Float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#Int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Two:]] = OpConstant %[[#Int]] 2
; CHECK-DAG: %[[#Array:]] = OpTypeArray %[[#Float]] %[[#Two]]
; CHECK-DAG: %[[#Struct:]] = OpTypeStruct %[[#Float]] %[[#Float]]

; CHECK: %[[#L:]] = OpLoad %[[#Array]]
; CHECK: OpCompositeExtract %[[#Float]] %[[#L]] 0
define spir_kernel void @freeze_loaded(ptr addrspace(1) %out, ptr addrspace(1) %pa) {
  %a = load [2 x float], ptr addrspace(1) %pa
  %v = freeze [2 x float] %a
  %e0 = extractvalue [2 x float] %v, 0
  store float %e0, ptr addrspace(1) %out
  ret void
}

; CHECK: %[[#Ins:]] = OpCompositeInsert %[[#Struct]] %[[#]] %[[#]] 1
; CHECK: OpCompositeExtract %[[#Float]] %[[#Ins]] 1
define spir_kernel void @freeze_insertvalue(ptr addrspace(1) %out, float %x) {
  %ins = insertvalue { float, float } zeroinitializer, float %x, 1
  %v = freeze { float, float } %ins
  %e1 = extractvalue { float, float } %v, 1
  store float %e1, ptr addrspace(1) %out
  ret void
}

; CHECK: %[[#S:]] = OpLoad %[[#Array]]
; CHECK: OpStore %[[#]] %[[#S]]
define spir_kernel void @freeze_store_direct(ptr addrspace(1) %out, ptr addrspace(1) %pa) {
  %a = load [2 x float], ptr addrspace(1) %pa
  %v = freeze [2 x float] %a
  store [2 x float] %v, ptr addrspace(1) %out
  ret void
}
