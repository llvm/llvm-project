; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpName %[[#WD:]] "__spirv_BuiltInWorkDim"
; CHECK-DAG: OpName %[[#GS:]] "__spirv_BuiltInGlobalSize"
; CHECK-DAG: OpName %[[#GII:]] "__spirv_BuiltInGlobalInvocationId"
; CHECK-DAG: OpName %[[#WS:]] "__spirv_BuiltInWorkgroupSize"
; CHECK-DAG: OpName %[[#EWS:]] "__spirv_BuiltInEnqueuedWorkgroupSize"
; CHECK-DAG: OpName %[[#LLI:]] "__spirv_BuiltInLocalInvocationId"
; CHECK-DAG: OpName %[[#NW:]] "__spirv_BuiltInNumWorkgroups"
; CHECK-DAG: OpName %[[#WI:]] "__spirv_BuiltInWorkgroupId"
; CHECK-DAG: OpName %[[#GO:]] "__spirv_BuiltInGlobalOffset"
; CHECK-DAG: OpName %[[#GLI:]] "__spirv_BuiltInGlobalLinearId"
; CHECK-DAG: OpName %[[#LLII:]] "__spirv_BuiltInLocalInvocationIndex"
; CHECK-DAG: OpName %[[#SS:]] "__spirv_BuiltInSubgroupSize"
; CHECK-DAG: OpName %[[#SMS:]] "__spirv_BuiltInSubgroupMaxSize"
; CHECK-DAG: OpName %[[#NS:]] "__spirv_BuiltInNumSubgroups"
; CHECK-DAG: OpName %[[#NES:]] "__spirv_BuiltInNumEnqueuedSubgroups"
; CHECK-DAG: OpName %[[#SI:]] "__spirv_BuiltInSubgroupId"
; CHECK-DAG: OpName %[[#SLII:]] "__spirv_BuiltInSubgroupLocalInvocationId"

; CHECK-DAG: OpDecorate %[[#NW]] BuiltIn NumWorkgroups
; CHECK-DAG: OpDecorate %[[#WS]] BuiltIn WorkgroupSize
; CHECK-DAG: OpDecorate %[[#WI]] BuiltIn WorkgroupId
; CHECK-DAG: OpDecorate %[[#LLI]] BuiltIn LocalInvocationId
; CHECK-DAG: OpDecorate %[[#GII]] BuiltIn GlobalInvocationId
; CHECK-DAG: OpDecorate %[[#LLII]] BuiltIn LocalInvocationIndex
; CHECK-DAG: OpDecorate %[[#WD]] BuiltIn WorkDim
; CHECK-DAG: OpDecorate %[[#GS]] BuiltIn GlobalSize
; CHECK-DAG: OpDecorate %[[#EWS]] BuiltIn EnqueuedWorkgroupSize
; CHECK-DAG: OpDecorate %[[#GO]] BuiltIn GlobalOffset
; CHECK-DAG: OpDecorate %[[#GLI]] BuiltIn GlobalLinearId
; CHECK-DAG: OpDecorate %[[#SS]] BuiltIn SubgroupSize
; CHECK-DAG: OpDecorate %[[#SMS]] BuiltIn SubgroupMaxSize
; CHECK-DAG: OpDecorate %[[#NS]] BuiltIn NumSubgroups
; CHECK-DAG: OpDecorate %[[#NES]] BuiltIn NumEnqueuedSubgroups
; CHECK-DAG: OpDecorate %[[#SI]] BuiltIn SubgroupId
; CHECK-DAG: OpDecorate %[[#SLII]] BuiltIn SubgroupLocalInvocationId

; CHECK-DAG: %[[#SizeT:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#Int32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#SizeTPtr:]] = OpTypePointer Input %[[#SizeT]]
; CHECK-DAG: %[[#Int32Ptr:]] = OpTypePointer Input %[[#Int32]]

; CHECK-DAG: %[[#GLI]] = OpVariable %[[#SizeTPtr]] Input
; CHECK-DAG: %[[#LLII]] = OpVariable %[[#SizeTPtr]] Input
; CHECK-DAG: %[[#WD]] = OpVariable %[[#Int32Ptr]] Input
; CHECK-DAG: %[[#SS]] = OpVariable %[[#Int32Ptr]] Input
; CHECK-DAG: %[[#SMS]] = OpVariable %[[#Int32Ptr]] Input
; CHECK-DAG: %[[#NS]] = OpVariable %[[#Int32Ptr]] Input
; CHECK-DAG: %[[#NES]] = OpVariable %[[#Int32Ptr]] Input
; CHECK-DAG: %[[#SI]] = OpVariable %[[#Int32Ptr]] Input
; CHECK-DAG: %[[#SLII]] = OpVariable %[[#Int32Ptr]] Input

; CHECK: OpFunction
; CHECK: %[[#]] = OpLoad %[[#SizeT]] %[[#GLI]]
; CHECK: %[[#]] = OpLoad %[[#SizeT]] %[[#LLII]]
; CHECK: %[[#]] = OpLoad %[[#Int32]] %[[#WD]]
; CHECK: %[[#]] = OpLoad %[[#Int32]] %[[#SS]]
; CHECK: %[[#]] = OpLoad %[[#Int32]] %[[#SMS]]
; CHECK: %[[#]] = OpLoad %[[#Int32]] %[[#NS]]
; CHECK: %[[#]] = OpLoad %[[#Int32]] %[[#NES]]
; CHECK: %[[#]] = OpLoad %[[#Int32]] %[[#SI]]
; CHECK: %[[#]] = OpLoad %[[#Int32]] %[[#SLII]]

@__spirv_BuiltInWorkDim = external addrspace(1) global i32
@__spirv_BuiltInGlobalSize = external addrspace(1) global <3 x i32>
@__spirv_BuiltInGlobalInvocationId = external addrspace(1) global <3 x i32>
@__spirv_BuiltInWorkgroupSize = external addrspace(1) global <3 x i32>
@__spirv_BuiltInEnqueuedWorkgroupSize = external addrspace(1) global <3 x i32>
@__spirv_BuiltInLocalInvocationId = external addrspace(1) global <3 x i32>
@__spirv_BuiltInNumWorkgroups = external addrspace(1) global <3 x i32>
@__spirv_BuiltInWorkgroupId = external addrspace(1) global <3 x i32>
@__spirv_BuiltInGlobalOffset = external addrspace(1) global <3 x i32>
@__spirv_BuiltInGlobalLinearId = external addrspace(1) global i32
@__spirv_BuiltInLocalInvocationIndex = external addrspace(1) global i32
@__spirv_BuiltInSubgroupSize = external addrspace(1) global i32
@__spirv_BuiltInSubgroupMaxSize = external addrspace(1) global i32
@__spirv_BuiltInNumSubgroups = external addrspace(1) global i32
@__spirv_BuiltInNumEnqueuedSubgroups = external addrspace(1) global i32
@__spirv_BuiltInSubgroupId = external addrspace(1) global i32
@__spirv_BuiltInSubgroupLocalInvocationId = external addrspace(1) global i32

@G_r1 = global i64 0
@G_r2 = global i64 0
@G_r3 = global i32 0
@G_r4 = global i32 0
@G_r5 = global i32 0
@G_r6 = global i32 0
@G_r7 = global i32 0
@G_r8 = global i32 0
@G_r9 = global i32 0

define spir_kernel void @_Z1wv() {
entry:
  %r1 = tail call spir_func i64 @get_global_linear_id()
  store i64 %r1, ptr @G_r1
  %r2 = tail call spir_func i64 @get_local_linear_id()
  store i64 %r2, ptr @G_r2
  %r3 = tail call spir_func i32 @get_work_dim()
  store i32 %r3, ptr @G_r3
  %r4 = tail call spir_func i32 @get_sub_group_size()
  store i32 %r4, ptr @G_r4
  %r5 = tail call spir_func i32 @get_max_sub_group_size()
  store i32 %r5, ptr @G_r5
  %r6 = tail call spir_func i32 @get_num_sub_groups()
  store i32 %r6, ptr @G_r6
  %r7 = tail call spir_func i32 @get_enqueued_num_sub_groups()
  store i32 %r7, ptr @G_r7
  %r8 = tail call spir_func i32 @get_sub_group_id()
  store i32 %r8, ptr @G_r8
  %r9 = tail call spir_func i32 @get_sub_group_local_id()
  store i32 %r9, ptr @G_r9
  ret void
}

declare spir_func i64 @get_global_linear_id()
declare spir_func i64 @get_local_linear_id()
declare spir_func i32 @get_work_dim()
declare spir_func i32 @get_sub_group_size()
declare spir_func i32 @get_max_sub_group_size()
declare spir_func i32 @get_num_sub_groups()
declare spir_func i32 @get_enqueued_num_sub_groups()
declare spir_func i32 @get_sub_group_id()
declare spir_func i32 @get_sub_group_local_id()
