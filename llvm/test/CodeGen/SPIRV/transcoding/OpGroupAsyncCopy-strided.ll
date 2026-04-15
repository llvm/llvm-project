; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#IntTy:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Int8Ty:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#EventTy:]] = OpTypeEvent
; CHECK-DAG: %[[#WGPtrTy:]] = OpTypePointer Workgroup %[[#Int8Ty]]
; CHECK-DAG: %[[#CWGPtrTy:]] = OpTypePointer CrossWorkgroup %[[#Int8Ty]]
; CHECK-DAG: %[[#Scope:]] = OpConstant %[[#IntTy]] 2
; CHECK-DAG: %[[#NumElem:]] = OpConstant %[[#IntTy]] 123
; CHECK-DAG: %[[#Stride:]] = OpConstant %[[#IntTy]] 1
; CHECK-DAG: %[[#DstNull:]] = OpConstantNull %[[#WGPtrTy]]
; CHECK-DAG: %[[#SrcNull:]] = OpConstantNull %[[#CWGPtrTy]]
; CHECK-DAG: %[[#EventNull:]] = OpConstantNull %[[#EventTy]]
; CHECK-DAG: %[[#GenPtrEventTy:]] = OpTypePointer Generic %[[#EventTy]]
; CHECK-DAG: %[[#FunPtrEventTy:]] = OpTypePointer Function %[[#EventTy]]
; CHECK: OpFunction
; CHECK: %[[#Var:]] = OpVariable %[[#FunPtrEventTy]] Function
; CHECK: %[[#ResEvent:]] = OpGroupAsyncCopy %[[#EventTy]] %[[#Scope]] %[[#DstNull]] %[[#SrcNull]] %[[#NumElem]] %[[#Stride]] %[[#EventNull]]
; CHECK: OpStore %[[#Var]] %[[#ResEvent]]
; CHECK: %[[#PtrEventGen:]] = OpPtrCastToGeneric %[[#GenPtrEventTy]] %[[#Var]]
; CHECK: OpGroupWaitEvents %[[#Scope]] %[[#Stride]] %[[#PtrEventGen]]
; CHECK: OpFunctionEnd

define spir_kernel void @foo() {
  %event = alloca target("spirv.Event"), align 8
  %call = call spir_func target("spirv.Event") @_Z29async_work_group_strided_copyPU3AS3hPU3AS1Khjj9ocl_event(ptr addrspace(3) null, ptr addrspace(1) null, i32 123, i32 1, target("spirv.Event") zeroinitializer)
  store target("spirv.Event") %call, ptr %event, align 8
  %event.ascast = addrspacecast ptr %event to ptr addrspace(4)
  call spir_func void @_Z17wait_group_eventsiPU3AS49ocl_event(i32 1, ptr addrspace(4) %event.ascast)
  ret void
}

declare spir_func target("spirv.Event") @_Z29async_work_group_strided_copyPU3AS3hPU3AS1Khjj9ocl_event(ptr addrspace(3), ptr addrspace(1), i32, i32, target("spirv.Event"))
declare spir_func void @_Z17wait_group_eventsiPU3AS49ocl_event(i32, ptr addrspace(4))
