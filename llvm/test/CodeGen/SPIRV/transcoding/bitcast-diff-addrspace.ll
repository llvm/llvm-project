; Check that bitcast for pointers implies that the address spaces must match.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s --spirv-ext=+SPV_INTEL_usm_storage_classes -o - | FileCheck %s
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s --spirv-ext=+SPV_INTEL_usm_storage_classes -o - -filetype=obj | spirv-val %}

; CHECK: Capability USMStorageClassesINTEL
; CHECK: OpExtension "SPV_INTEL_usm_storage_classes"
; CHECK-DAG: OpName %[[#Bar:]] "bar"
; CHECK-DAG: %[[#Void:]] = OpTypeVoid
; CHECK-DAG: %[[#Char:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#GenPtrChar:]] = OpTypePointer Generic %[[#Char]]
; CHECK-DAG: %[[#AtDevice:]] = OpTypePointer DeviceOnlyINTEL %[[#Char]]
; CHECK-DAG: %[[#AtHost:]] = OpTypePointer HostOnlyINTEL %[[#Char]]
; CHECK-DAG: %[[#PtrVarHost:]] = OpTypePointer Function %[[#AtHost]]
; CHECK-DAG: %[[#PtrVarDevice:]] = OpTypePointer Function %[[#AtDevice]]
; CHECK: OpFunction
; CHECK: %[[#VarDevice:]] = OpVariable %[[#PtrVarDevice]] Function
; CHECK: %[[#VarHost:]] = OpVariable %[[#PtrVarHost]] Function
; CHECK: %[[#LoadedDevice:]] = OpLoad %[[#AtDevice]] %[[#VarDevice]]
; CHECK: %[[#CastedFromDevice:]] = OpPtrCastToGeneric %[[#GenPtrChar]] %[[#LoadedDevice]]
; CHECK: OpFunctionCall %[[#Void]] %[[#Bar]] %[[#CastedFromDevice]]
; CHECK: %[[#LoadedHost:]] = OpLoad %[[#AtHost]] %[[#VarHost]]
; CHECK: %[[#CastedFromHost:]] = OpPtrCastToGeneric %[[#GenPtrChar]] %[[#LoadedHost]]
; CHECK: OpFunctionCall %[[#Void]] %[[#Bar]] %[[#CastedFromHost]]

define spir_func void @foo() {
entry:
  %device_var = alloca ptr addrspace(5)
  %host_var = alloca ptr addrspace(6)
  %p1 = load ptr addrspace(5), ptr %device_var
  %p2 = addrspacecast ptr addrspace(5) %p1 to ptr addrspace(4)
  call spir_func void @bar(ptr addrspace(4) %p2)
  %p3 = load ptr addrspace(6), ptr %host_var
  %p4 = addrspacecast ptr addrspace(6) %p3 to ptr addrspace(4)
  call spir_func void @bar(ptr addrspace(4) %p4)
  ret void
}

define spir_func void @bar(ptr addrspace(4) %data) {
entry:
  %data.addr = alloca ptr addrspace(4)
  store ptr addrspace(4) %data, ptr %data.addr
  ret void
}
