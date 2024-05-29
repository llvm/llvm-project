; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: %[[#EventTy:]] = OpTypeEvent
; CHECK: %[[#StructEventTy:]] = OpTypeStruct %[[#EventTy]]
; CHECK: %[[#GenPtrStructEventTy:]] = OpTypePointer Generic %[[#StructEventTy]]
; CHECK: %[[#FunPtrStructEventTy:]] = OpTypePointer Function %[[#StructEventTy]]
; CHECK: %[[#GenPtrEventTy:]] = OpTypePointer Generic %[[#EventTy:]]
; CHECK: OpFunction
; CHECK: %[[#Var:]] = OpVariable %[[#FunPtrStructEventTy]] Function
; CHECK-NEXT: %[[#AddrspacecastVar:]] = OpPtrCastToGeneric %[[#GenPtrStructEventTy]] %[[#Var]]
; CHECK-NEXT: %[[#BitcastVar:]] = OpBitcast %[[#GenPtrEventTy]] %[[#AddrspacecastVar]]
; CHECK-NEXT: OpGroupWaitEvents %[[#]] %[[#]] %[[#BitcastVar]]

%"class.sycl::_V1::device_event" = type { target("spirv.Event") }

define weak_odr dso_local spir_kernel void @foo() {
entry:
  %var = alloca %"class.sycl::_V1::device_event"
  %eventptr = addrspacecast ptr %var to ptr addrspace(4)
  call spir_func void @_Z23__spirv_GroupWaitEventsjiP9ocl_event(i32 2, i32 1, ptr addrspace(4) %eventptr)
  ret void
}

declare dso_local spir_func void @_Z23__spirv_GroupWaitEventsjiP9ocl_event(i32, i32, ptr addrspace(4))
