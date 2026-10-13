; RUN: split-file %s %t
; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %t/caller.ll -o - | FileCheck %s
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %t/caller.ll -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %t/caller.ll -filetype=obj -o %t/caller32.spv %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %t/callee.ll -filetype=obj -o %t/callee32.spv %}
; RUN: %if spirv-tools %{ spirv-link %t/caller32.spv %t/callee32.spv -o %t/linked32.spv %}
; RUN: %if spirv-tools %{ spirv-val %t/linked32.spv %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %t/caller.ll -filetype=obj -o %t/caller64.spv %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %t/callee.ll -filetype=obj -o %t/callee64.spv %}
; RUN: %if spirv-tools %{ spirv-link %t/caller64.spv %t/callee64.spv -o %t/linked64.spv %}
; RUN: %if spirv-tools %{ spirv-val %t/linked64.spv %}

; Infer a namespaced free function's parameter independently of its definition.
; Both callers must use int*, including the one passing a float allocation.
; CHECK-DAG: OpName %[[#Store:]] "_ZN2ns5storeEPi"
; CHECK-DAG: OpName %[[#TestFloat:]] "test_float"
; CHECK-DAG: OpName %[[#TestInt:]] "test_int"
; CHECK-DAG: %[[#Void:]] = OpTypeVoid
; CHECK-DAG: %[[#Int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#FloatPtr:]] = OpTypePointer Function %[[#Float]]
; CHECK-DAG: %[[#IntPtr:]] = OpTypePointer Function %[[#Int]]
; CHECK-DAG: %[[#GenericIntPtr:]] = OpTypePointer Generic %[[#Int]]
; CHECK-DAG: %[[#StoreTy:]] = OpTypeFunction %[[#Void]] %[[#GenericIntPtr]]
; CHECK: %[[#Store]] = OpFunction %[[#Void]] None %[[#StoreTy]]
; CHECK: OpFunctionParameter %[[#GenericIntPtr]]
; CHECK: %[[#TestFloat]] = OpFunction
; CHECK: %[[#F:]] = OpVariable %[[#FloatPtr]] Function
; CHECK: %[[#FI:]] = OpBitcast %[[#IntPtr]] %[[#F]]
; CHECK: %[[#FG:]] = OpPtrCastToGeneric %[[#GenericIntPtr]] %[[#FI]]
; CHECK: OpFunctionCall %[[#Void]] %[[#Store]] %[[#FG]]
; CHECK: %[[#TestInt]] = OpFunction
; CHECK: %[[#I:]] = OpVariable %[[#IntPtr]] Function
; CHECK: %[[#IG:]] = OpPtrCastToGeneric %[[#GenericIntPtr]] %[[#I]]
; CHECK: OpFunctionCall %[[#Void]] %[[#Store]] %[[#IG]]

;--- caller.ll
declare spir_func void @_ZN2ns5storeEPi(ptr addrspace(4))

define spir_kernel void @test_float() {
  %p = alloca float, align 4
  store float 1.0, ptr %p, align 4
  %g = addrspacecast ptr %p to ptr addrspace(4)
  call spir_func void @_ZN2ns5storeEPi(ptr addrspace(4) %g)
  ret void
}

define spir_kernel void @test_int() {
  %p = alloca i32, align 4
  %g = addrspacecast ptr %p to ptr addrspace(4)
  call spir_func void @_ZN2ns5storeEPi(ptr addrspace(4) %g)
  ret void
}

;--- callee.ll
define spir_func void @_ZN2ns5storeEPi(ptr addrspace(4) %p) {
  store i32 42, ptr addrspace(4) %p, align 4
  ret void
}
