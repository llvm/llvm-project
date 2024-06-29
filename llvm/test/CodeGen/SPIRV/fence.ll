; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpName %[[#GetScope:]] "_Z8getScopev"
; CHECK-DAG: %[[#Long:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#ScopeDevice:]] = OpConstant %[[#Long]] 1
; CHECK-DAG: %[[#WrkGrpConst2:]] = OpConstant %[[#Long]] 2
; CHECK-DAG: %[[#Const3:]] = OpConstant %[[#Long]] 3
; CHECK-DAG: %[[#InvocationConst4:]] = OpConstant %[[#Long]] 4
; CHECK-DAG: %[[#Const8:]] = OpConstant %[[#Long]] 8
; CHECK-DAG: %[[#Const16:]] = OpConstant %[[#Long]] 16
; CHECK-DAG: %[[#Const912:]] = OpConstant %[[#Long]] 912
; CHECK: OpMemoryBarrier %[[#ScopeDevice]] %[[#WrkGrpConst2]]
; CHECK: OpMemoryBarrier %[[#ScopeDevice]] %[[#InvocationConst4]]
; CHECK: OpMemoryBarrier %[[#ScopeDevice]] %[[#Const8]]
; CHECK: OpMemoryBarrier %[[#InvocationConst4]] %[[#Const16]]
; CHECK: OpMemoryBarrier %[[#WrkGrpConst2]] %[[#InvocationConst4]]
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

define spir_func void @barrier_test1() {
  %scope = call noundef i32 @_Z8getScopev()
  call void @_Z22__spirv_ControlBarrieriii(i32 noundef 3, i32 noundef %scope, i32 noundef 912)
  ret void
}

declare spir_func void @_Z22__spirv_ControlBarrieriii(i32 noundef, i32 noundef, i32 noundef)
declare spir_func i32 @_Z8getScopev()
