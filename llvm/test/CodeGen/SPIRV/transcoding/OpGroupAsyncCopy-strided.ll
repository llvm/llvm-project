; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV-DAG: %[[#IntTy:]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG: %[[#Int8Ty:]] = OpTypeInt 8 0
; CHECK-SPIRV-DAG: %[[#EventTy:]] = OpTypeEvent
; CHECK-SPIRV-DAG: %[[#WGPtrTy:]] = OpTypePointer Workgroup %[[#Int8Ty]]
; CHECK-SPIRV-DAG: %[[#CWGPtrTy:]] = OpTypePointer CrossWorkgroup %[[#Int8Ty]]
; CHECK-SPIRV-DAG: %[[#Scope:]] = OpConstant %[[#IntTy]] 2
; CHECK-SPIRV-DAG: %[[#NumElem:]] = OpConstant %[[#IntTy]] 123
; CHECK-SPIRV-DAG: %[[#Stride:]] = OpConstant %[[#IntTy]] 1
; CHECK-SPIRV-DAG: %[[#DstNull:]] = OpConstantNull %[[#WGPtrTy]]
; CHECK-SPIRV-DAG: %[[#SrcNull:]] = OpConstantNull %[[#CWGPtrTy]]
; CHECK-SPIRV-DAG: %[[#EventNull:]] = OpConstantNull %[[#EventTy]]
; CHECK-SPIRV-DAG: %[[#GenPtrEventTy:]] = OpTypePointer Generic %[[#EventTy]]
; CHECK-SPIRV-DAG: %[[#FunPtrEventTy:]] = OpTypePointer Function %[[#EventTy]]
; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#Var:]] = OpVariable %[[#FunPtrEventTy]] Function
; CHECK-SPIRV: %[[#ResEvent:]] = OpGroupAsyncCopy %[[#EventTy]] %[[#Scope]] %[[#DstNull]] %[[#SrcNull]] %[[#NumElem]] %[[#Stride]] %[[#EventNull]]
; CHECK-SPIRV: OpStore %[[#Var]] %[[#ResEvent]]
; CHECK-SPIRV: %[[#PtrEventGen:]] = OpPtrCastToGeneric %[[#GenPtrEventTy]] %[[#Var]]
; CHECK-SPIRV: OpGroupWaitEvents %[[#Scope]] %[[#Stride]] %[[#PtrEventGen]]
; CHECK-SPIRV: OpFunctionEnd

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
