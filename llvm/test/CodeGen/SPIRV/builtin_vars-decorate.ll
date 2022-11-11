; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s

; CHECK: OpName %[[#WD:]] "__spirv_BuiltInWorkDim"
; CHECK: OpName %[[#GS:]] "__spirv_BuiltInGlobalSize"
; CHECK: OpName %[[#GII:]] "__spirv_BuiltInGlobalInvocationId"
; CHECK: OpName %[[#WS:]] "__spirv_BuiltInWorkgroupSize"
; CHECK: OpName %[[#EWS:]] "__spirv_BuiltInEnqueuedWorkgroupSize"
; CHECK: OpName %[[#LLI:]] "__spirv_BuiltInLocalInvocationId"
; CHECK: OpName %[[#NW:]] "__spirv_BuiltInNumWorkgroups"
; CHECK: OpName %[[#WI:]] "__spirv_BuiltInWorkgroupId"
; CHECK: OpName %[[#GO:]] "__spirv_BuiltInGlobalOffset"
; CHECK: OpName %[[#GLI:]] "__spirv_BuiltInGlobalLinearId"
; CHECK: OpName %[[#LLII:]] "__spirv_BuiltInLocalInvocationIndex"
; CHECK: OpName %[[#SS:]] "__spirv_BuiltInSubgroupSize"
; CHECK: OpName %[[#SMS:]] "__spirv_BuiltInSubgroupMaxSize"
; CHECK: OpName %[[#NS:]] "__spirv_BuiltInNumSubgroups"
; CHECK: OpName %[[#NES:]] "__spirv_BuiltInNumEnqueuedSubgroups"
; CHECK: OpName %[[#SI:]] "__spirv_BuiltInSubgroupId"
; CHECK: OpName %[[#SLII:]] "__spirv_BuiltInSubgroupLocalInvocationId"

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

define spir_kernel void @_Z1wv() {
entry:
  ret void
}
