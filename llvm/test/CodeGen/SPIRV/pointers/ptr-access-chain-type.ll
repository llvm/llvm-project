; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#Char:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#Long:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#CharPtr:]] = OpTypePointer CrossWorkgroup %[[#Char]]
; CHECK-DAG: %[[#LongPtr:]] = OpTypePointer CrossWorkgroup %[[#Long]]
; CHECK-DAG: %[[#LongPtrWG:]] = OpTypePointer Workgroup %[[#Long]]
; CHECK: OpFunction
; CHECK: OpFunctionParameter
; CHECK: %[[#Dest:]] = OpFunctionParameter %[[#CharPtr]]
; CHECK: %[[#Src:]] = OpFunctionParameter %[[#LongPtrWG]]
; CHECK: %[[#InDest:]] = OpInBoundsPtrAccessChain %[[#CharPtr]] %[[#Dest]] %[[#]]
; CHECK: %[[#InDestCasted:]] = OpBitcast %[[#LongPtr]] %[[#InDest]]
; CHECK: OpGroupAsyncCopy %[[#]] %[[#]] %[[#InDestCasted]] %[[#Src]] %[[#]] %[[#]] %[[#]]

define spir_kernel void @foo(i64 %idx, ptr addrspace(1) %dest, ptr addrspace(3) %src) {
  %p1 = getelementptr inbounds i8, ptr addrspace(1) %dest, i64 %idx
  %res = tail call spir_func target("spirv.Event") @_Z22__spirv_GroupAsyncCopyjPU3AS1iPU3AS3Kimm9ocl_event(i32 2, ptr addrspace(1) %p1, ptr addrspace(3) %src, i64 128, i64 1, target("spirv.Event") zeroinitializer)
  ret void
}

; For this test case the mangling is important.
declare dso_local spir_func target("spirv.Event") @_Z22__spirv_GroupAsyncCopyjPU3AS1iPU3AS3Kimm9ocl_event(i32, ptr addrspace(1), ptr addrspace(3), i64, i64, target("spirv.Event"))
