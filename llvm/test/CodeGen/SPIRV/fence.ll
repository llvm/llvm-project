; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpName %[[#GetScope:]] "_Z8getScopev"
; CHECK-DAG: %[[#Long:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#ScopeWorkgroup:]] = OpConstant %[[#Long]] 2{{$}}
; CHECK-DAG: %[[#ScopeAllSvmDevices:]] = OpConstantNull %[[#Long]]
; CHECK-DAG: %[[#ScopeInvocation:]] = OpConstant %[[#Long]] 4{{$}}
; CHECK-DAG: %[[#Acquire:]] = OpConstant %[[#Long]] 2818
; CHECK-DAG: %[[#Release:]] = OpConstant %[[#Long]] 2820
; CHECK-DAG: %[[#AcqRel:]] = OpConstant %[[#Long]] 2824
; CHECK-DAG: %[[#SeqCst:]] = OpConstant %[[#Long]] 2832
; CHECK-DAG: %[[#Const3:]] = OpConstant %[[#Long]] 3{{$}}
; CHECK-DAG: %[[#Const912:]] = OpConstant %[[#Long]] 912
; CHECK-DAG: %[[#Const42:]] = OpConstant %[[#Long]] 42
; CHECK-DAG: %[[#Const1:]] = OpConstant %[[#Long]] 1{{$}}
; CHECK-DAG: %[[#RelaxedCW:]] = OpConstant %[[#Long]] 512
; CHECK: OpMemoryBarrier %[[#ScopeAllSvmDevices]] %[[#Acquire]]
; CHECK: OpMemoryBarrier %[[#ScopeAllSvmDevices]] %[[#Release]]
; CHECK: OpMemoryBarrier %[[#ScopeAllSvmDevices]] %[[#AcqRel]]
; CHECK: OpMemoryBarrier %[[#ScopeInvocation]] %[[#SeqCst]]
; CHECK: OpMemoryBarrier %[[#ScopeWorkgroup]] %[[#Release]]
; CHECK: OpFunctionEnd
; CHECK: OpStore %[[#]] %[[#Const42]]
; CHECK: OpMemoryBarrier %[[#ScopeAllSvmDevices]] %[[#Release]]
; CHECK: OpAtomicStore %[[#]] %[[#ScopeAllSvmDevices]] %[[#RelaxedCW]] %[[#Const1]]
; CHECK: OpFunctionEnd
; CHECK: %[[#ScopeId:]] = OpFunctionCall %[[#Long]] %[[#GetScope]]
; CHECK: OpControlBarrier %[[#Const3]] %[[#ScopeId:]] %[[#Const912]]

define spir_kernel void @fence_test_kernel1(ptr addrspace(1) noalias %s.ascast) {
  fence acquire
  ret void
}

define spir_kernel void @fence_test_kernel2(ptr addrspace(1) noalias %s.ascast) {
  fence release
  ret void
}

define spir_kernel void @fence_test_kernel3(ptr addrspace(1) noalias %s.ascast) {
  fence acq_rel
  ret void
}

define spir_kernel void @fence_test_kernel4(ptr addrspace(1) noalias %s.ascast) {
  fence syncscope("singlethread") seq_cst
  ret void
}

define spir_kernel void @fence_test_kernel5(ptr addrspace(1) noalias %s.ascast) {
  fence syncscope("workgroup") release
  ret void
}

define spir_kernel void @fence_test_publish(ptr addrspace(1) %data, ptr addrspace(1) %flag) {
  store i32 42, ptr addrspace(1) %data, align 4
  fence release
  store atomic i32 1, ptr addrspace(1) %flag monotonic, align 4
  ret void
}

define spir_func void @barrier_test1() {
  %scope = call noundef i32 @_Z8getScopev()
  call void @_Z22__spirv_ControlBarrieriii(i32 noundef 3, i32 noundef %scope, i32 noundef 912)
  ret void
}

declare spir_func void @_Z22__spirv_ControlBarrieriii(i32 noundef, i32 noundef, i32 noundef)
declare spir_func i32 @_Z8getScopev()
