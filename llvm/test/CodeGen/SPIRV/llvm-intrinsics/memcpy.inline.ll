; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck -implicit-check-not=OpFunctionCall %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#Int8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#Int32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Int64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#Ptr_CrossWG_8:]] = OpTypePointer CrossWorkgroup %[[#Int8]]
; CHECK-DAG: %[[#Const_64:]] = OpConstant %[[#Int32]] 64
; CHECK-DAG: %[[#Const_30:]] = OpConstant %[[#Int32]] 30

; CHECK: %[[#Param1:]] = OpFunctionParameter %[[#Ptr_CrossWG_8]]
; CHECK: %[[#Param2:]] = OpFunctionParameter %[[#Ptr_CrossWG_8]]
; CHECK: %[[#Size1:]] = OpUConvert %[[#Int64]] %[[#Const_64]]
; CHECK: OpCopyMemorySized %[[#Param2]] %[[#Param1]] %[[#Size1]] Aligned 64

; CHECK: %[[#Param3:]] = OpFunctionParameter %[[#Ptr_CrossWG_8]]
; CHECK: %[[#Param4:]] = OpFunctionParameter %[[#Ptr_CrossWG_8]]
; CHECK: %[[#Size2:]] = OpUConvert %[[#Int64]] %[[#Const_30]]
; CHECK: OpCopyMemorySized %[[#Param4]] %[[#Param3]] %[[#Size2]] Aligned 1

define spir_kernel void @test_full_copy(ptr addrspace(1) captures(none) readonly %in, ptr addrspace(1) captures(none) %out) {
  call void @llvm.memcpy.inline.p1.p1.i32(ptr addrspace(1) align 64 %out, ptr addrspace(1) align 64 %in, i32 64, i1 false)
  ret void
}

define spir_kernel void @test_array(ptr addrspace(1) %in, ptr addrspace(1) %out) {
  call void @llvm.memcpy.inline.p1.p1.i32(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 30, i1 false)
  ret void
}

declare void @llvm.memcpy.inline.p1.p1.i32(ptr addrspace(1) captures(none), ptr addrspace(1) captures(none) readonly, i32 immarg, i1 immarg)
