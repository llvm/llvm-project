; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#TyEvent:]] = OpTypeEvent
; CHECK-DAG: %[[#TyStruct:]] = OpTypeStruct %[[#TyEvent]]
; CHECK-DAG: %[[#ConstEvent:]] = OpConstantNull %[[#TyEvent]]
; CHECK-DAG: %[[#TyEventPtr:]] = OpTypePointer Function %[[#TyEvent]]
; CHECK-DAG: %[[#TyStructPtr:]] = OpTypePointer Function %[[#TyStruct]]
; CHECK: OpFunction
; CHECK: OpFunctionParameter
; CHECK: %[[#Src:]] = OpFunctionParameter
; CHECK: OpVariable %[[#TyStructPtr]] Function
; CHECK: %[[#EventVar:]] = OpVariable %[[#TyEventPtr]] Function
; CHECK: %[[#Dest:]] = OpInBoundsPtrAccessChain
; CHECK: %[[#CopyRes:]] = OpGroupAsyncCopy %[[#TyEvent]] %[[#]] %[[#Dest]] %[[#Src]] %[[#]] %[[#]] %[[#ConstEvent]]
; CHECK: OpStore %[[#EventVar]] %[[#CopyRes]]

%"class.sycl::_V1::device_event" = type { target("spirv.Event") }

define spir_kernel void @foo(ptr addrspace(1) %_arg_out_ptr, ptr addrspace(3) noundef %_arg_local_acc) {
entry:
  %var = alloca %"class.sycl::_V1::device_event"
  %dev_event.i.sroa.0 = alloca target("spirv.Event")
  %add.ptr.i26 = getelementptr inbounds i32, ptr addrspace(1) %_arg_out_ptr, i64 0
  %call3.i = tail call spir_func target("spirv.Event") @_Z22__spirv_GroupAsyncCopyjPU3AS1iPU3AS3Kimm9ocl_event(i32 2, ptr addrspace(1) %add.ptr.i26, ptr addrspace(3) %_arg_local_acc, i64 16, i64 10, target("spirv.Event") zeroinitializer)
  store target("spirv.Event") %call3.i, ptr %dev_event.i.sroa.0
  ret void
}

declare dso_local spir_func target("spirv.Event") @_Z22__spirv_GroupAsyncCopyjPU3AS1iPU3AS3Kimm9ocl_event(i32, ptr addrspace(1), ptr addrspace(3), i64, i64, target("spirv.Event"))
