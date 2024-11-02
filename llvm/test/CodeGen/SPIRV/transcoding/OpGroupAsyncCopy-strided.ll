; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV-DAG: %[[#LongTy:]] = OpTypeInt 64 0
; CHECK-SPIRV-DAG: %[[#IntTy:]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG: %[[#EventTy:]] = OpTypeEvent
; CHECK-SPIRV-DAG: %[[#Scope:]] = OpConstant %[[#IntTy]] 2
; CHECK-SPIRV-DAG: %[[#Num:]] = OpConstant %[[#LongTy]] 123
; CHECK-SPIRV-DAG: %[[#Null:]] = OpConstantNull
; CHECK-SPIRV-DAG: %[[#Stride:]] = OpConstant %[[#LongTy]] 1
; CHECK-SPIRV-DAG: %[[#GenPtrEventTy:]] = OpTypePointer Generic %[[#EventTy]]
; CHECK-SPIRV-DAG: %[[#FunPtrEventTy:]] = OpTypePointer Function %[[#EventTy]]
; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#Var:]] = OpVariable %[[#]] Function
; CHECK-SPIRV: %[[#ResEvent:]] = OpGroupAsyncCopy %[[#EventTy]] %[[#Scope]] %[[#Null]] %[[#Null]] %[[#Num]] %[[#Stride]] %[[#Null]]
; CHECK-SPIRV: %[[#VarPtrEvent:]] = OpBitcast %[[#FunPtrEventTy]] %[[#Var]]
; CHECK-SPIRV: OpStore %[[#VarPtrEvent]] %[[#ResEvent]]
; CHECK-SPIRV: %[[#VarPtrEvent2:]] = OpBitcast %[[#FunPtrEventTy]] %[[#Var]]
; CHECK-SPIRV: %[[#PtrEventGen:]] = OpPtrCastToGeneric %[[#]] %[[#VarPtrEvent2]]
; CHECK-SPIRV: OpGroupWaitEvents %[[#Scope]] %[[#Num]] %[[#PtrEventGen]]
; CHECK-SPIRV: OpFunctionEnd

define spir_kernel void @foo() {
  %event = alloca ptr, align 8
  %call = call spir_func ptr @_Z29async_work_group_strided_copyPU3AS3hPU3AS1Khmm9ocl_event(ptr null, ptr null, i64 123, i64 1, ptr null)
  store ptr %call, ptr %event, align 8
  %event.ascast = addrspacecast ptr %event to ptr addrspace(4)
  call spir_func void @_Z17wait_group_eventsiPU3AS49ocl_event(i64 123, ptr addrspace(4) %event.ascast)
  ret void
}

declare spir_func ptr @_Z29async_work_group_strided_copyPU3AS3hPU3AS1Khmm9ocl_event(ptr, ptr, i64, i64, ptr)
declare spir_func void @_Z17wait_group_eventsiPU3AS49ocl_event(i64, ptr addrspace(4))
