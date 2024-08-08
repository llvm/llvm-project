; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#TyEvent:]] = OpTypeEvent
; CHECK-DAG: %[[#TyStruct:]] = OpTypeStruct %[[#TyEvent]]
; CHECK-DAG: %[[#ConstEvent:]] = OpConstantNull %[[#TyEvent]]
; CHECK-DAG: %[[#TyEventPtr:]] = OpTypePointer Function %[[#TyEvent]]
; CHECK-DAG: %[[#TyEventPtrGen:]] = OpTypePointer Generic %[[#TyEvent]]
; CHECK-DAG: %[[#TyStructPtr:]] = OpTypePointer Function %[[#TyStruct]]
; CHECK-DAG: %[[#TyChar:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#TyV4:]] = OpTypeVector %[[#TyChar]] 4
; CHECK-DAG: %[[#TyStructV4:]] = OpTypeStruct %[[#TyV4]]
; CHECK-DAG: %[[#TyPtrSV4_W:]] = OpTypePointer Workgroup %[[#TyStructV4]]
; CHECK-DAG: %[[#TyPtrSV4_CW:]] = OpTypePointer CrossWorkgroup %[[#TyStructV4]]
; CHECK-DAG: %[[#TyPtrV4_W:]] = OpTypePointer Workgroup %[[#TyV4]]
; CHECK-DAG: %[[#TyPtrV4_CW:]] = OpTypePointer CrossWorkgroup %[[#TyV4]]

; Check correct translation of __spirv_GroupAsyncCopy and target("spirv.Event") zeroinitializer

; CHECK: OpFunction
; CHECK: OpFunctionParameter
; CHECK: %[[#Src:]] = OpFunctionParameter
; CHECK: OpVariable %[[#TyStructPtr]] Function
; CHECK: %[[#EventVar:]] = OpVariable %[[#TyEventPtr]] Function
; CHECK: %[[#Dest:]] = OpInBoundsPtrAccessChain
; CHECK: %[[#CopyRes:]] = OpGroupAsyncCopy %[[#TyEvent]] %[[#]] %[[#Dest]] %[[#Src]] %[[#]] %[[#]] %[[#ConstEvent]]
; CHECK: OpStore %[[#EventVar]] %[[#CopyRes]]
; CHECK: OpFunctionEnd

%StructEvent = type { target("spirv.Event") }

define spir_kernel void @foo(ptr addrspace(1) %_arg_out_ptr, ptr addrspace(3) %_arg_local_acc) {
entry:
  %var = alloca %StructEvent
  %dev_event.i.sroa.0 = alloca target("spirv.Event")
  %add.ptr.i26 = getelementptr inbounds i32, ptr addrspace(1) %_arg_out_ptr, i64 0
  %call3.i = tail call spir_func target("spirv.Event") @_Z22__spirv_GroupAsyncCopyjPU3AS1iPU3AS3Kimm9ocl_event(i32 2, ptr addrspace(1) %add.ptr.i26, ptr addrspace(3) %_arg_local_acc, i64 16, i64 10, target("spirv.Event") zeroinitializer)
  store target("spirv.Event") %call3.i, ptr %dev_event.i.sroa.0
  ret void
}

declare dso_local spir_func target("spirv.Event") @_Z22__spirv_GroupAsyncCopyjPU3AS1iPU3AS3Kimm9ocl_event(i32, ptr addrspace(1), ptr addrspace(3), i64, i64, target("spirv.Event"))

; Check correct type inference when calling __spirv_GroupAsyncCopy:
; we expect that the Backend is able to deduce a type of the %_arg_Local
; given facts that it's possible to deduce a type of the %_arg
; and %_arg_Local and %_arg are source/destination arguments in OpGroupAsyncCopy

; CHECK: OpFunction
; CHECK: %[[#BarArg1:]] = OpFunctionParameter %[[#TyPtrSV4_W]]
; CHECK: %[[#BarArg2:]] = OpFunctionParameter %[[#TyPtrSV4_CW]]
; CHECK: %[[#EventVarBar:]] = OpVariable %[[#TyStructPtr]] Function
; CHECK: %[[#EventVarBarCasted2:]] = OpBitcast %[[#TyEventPtr]] %[[#EventVarBar]]
; CHECK: %[[#SrcBar:]] = OpInBoundsPtrAccessChain %[[#TyPtrSV4_CW]] %[[#BarArg2]] %[[#]]
; CHECK-DAG: %[[#BarArg1Casted:]] = OpBitcast %[[#TyPtrV4_W]] %[[#BarArg1]]
; CHECK-DAG: %[[#SrcBarCasted:]] = OpBitcast %[[#TyPtrV4_CW]] %[[#SrcBar]]
; CHECK: %[[#ResBar:]] = OpGroupAsyncCopy %[[#TyEvent]] %[[#]] %[[#BarArg1Casted]] %[[#SrcBarCasted]] %[[#]] %[[#]] %[[#ConstEvent]]
; CHECK: %[[#EventVarBarCasted:]] = OpBitcast %[[#TyEventPtr]] %[[#EventVarBar]]
; CHECK: OpStore %[[#EventVarBarCasted]] %[[#ResBar]]
; CHECK: %[[#EventVarBarGen:]] = OpPtrCastToGeneric %[[#TyEventPtrGen]] %[[#EventVarBarCasted2]]
; CHECK: OpGroupWaitEvents %[[#]] %[[#]] %[[#EventVarBarGen]]
; CHECK: OpFunctionEnd

%Vec4 = type { <4 x i8> }

define spir_kernel void @bar(ptr addrspace(3) %_arg_Local, ptr addrspace(1) readonly %_arg) {
entry:
  %E1 = alloca %StructEvent
  %srcptr = getelementptr inbounds %Vec4, ptr addrspace(1) %_arg, i64 0
  %r1 = tail call spir_func target("spirv.Event") @_Z22__spirv_GroupAsyncCopyjPU3AS3Dv4_aPU3AS1KS_mm9ocl_event(i32 2, ptr addrspace(3) %_arg_Local, ptr addrspace(1) %srcptr, i64 16, i64 10, target("spirv.Event") zeroinitializer)
  store target("spirv.Event") %r1, ptr %E1
  %E.ascast.i = addrspacecast ptr %E1 to ptr addrspace(4)
  call spir_func void @_Z23__spirv_GroupWaitEventsjiP9ocl_event(i32 2, i32 1, ptr addrspace(4) %E.ascast.i)
  ret void
}

declare dso_local spir_func target("spirv.Event") @_Z22__spirv_GroupAsyncCopyjPU3AS3Dv4_aPU3AS1KS_mm9ocl_event(i32, ptr addrspace(3), ptr addrspace(1), i64, i64, target("spirv.Event"))
declare dso_local spir_func void @_Z23__spirv_GroupWaitEventsjiP9ocl_event(i32, i32, ptr addrspace(4))
