; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV-DAG: %[[#Char:]] = OpTypeInt 8 0
; CHECK-SPIRV-DAG: %[[#GlobalCharPtr:]] = OpTypePointer CrossWorkgroup %[[#Char]]
; CHECK-SPIRV-DAG: %[[#LocalCharPtr:]] = OpTypePointer Workgroup %[[#Char]]
; CHECK-SPIRV-DAG: %[[#GenericCharPtr:]] = OpTypePointer Generic %[[#Char]]
; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#Arg1:]] = OpFunctionParameter %[[#GlobalCharPtr]]
; CHECK-SPIRV: %[[#Ptr1:]] = OpPtrCastToGeneric %[[#GenericCharPtr]] %[[#Arg1]]
; CHECK-SPIRV: OpGenericCastToPtr %[[#LocalCharPtr]] %[[#Ptr1]]
; CHECK-SPIRV: OpFunctionEnd
; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#Arg2:]] = OpFunctionParameter %[[#GlobalCharPtr]]
; CHECK-SPIRV: %[[#Ptr2:]] = OpPtrCastToGeneric %[[#GenericCharPtr]] %[[#Arg2]]
; CHECK-SPIRV: OpGenericCastToPtr %[[#LocalCharPtr]] %[[#Ptr2]]
; CHECK-SPIRV: OpFunctionEnd

define spir_kernel void @foo(ptr addrspace(1) %arg) {
entry:
  %p = addrspacecast ptr addrspace(1) %arg to ptr addrspace(3)
  ret void
}

define spir_kernel void @bar(ptr addrspace(1) %arg) {
entry:
  %p1 = addrspacecast ptr addrspace(1) %arg to ptr addrspace(4)
  %p2 = addrspacecast ptr addrspace(4) %p1 to ptr addrspace(3)
  ret void
}
