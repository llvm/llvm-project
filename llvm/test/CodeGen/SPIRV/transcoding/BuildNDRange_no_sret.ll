; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

%struct.ndrange_t = type { i32, [3 x i64], [3 x i64], [3 x i64] }

@__const.GS_3Dc = private unnamed_addr addrspace(2) constant [3 x i64] [i64 1, i64 4, i64 7], align 8
@__const.LS_3Dc = private unnamed_addr addrspace(2) constant [3 x i64] [i64 2, i64 5, i64 8], align 8
@__const.GO_3Dc = private unnamed_addr addrspace(2) constant [3 x i64] [i64 3, i64 6, i64 9], align 8

define spir_func void @test_ndrange_2D3D() local_unnamed_addr #0 {

; CHECK-LABEL: OpCapability Kernel
; CHECK: OpName %[[#GSName:]] "__const.GS_3Dc"
; CHECK: OpName %[[#LSName:]] "__const.LS_3Dc"
; CHECK: OpName %[[#GOName:]] "__const.GO_3Dc"

; CHECK: %[[#typeInt32:]] = OpTypeInt 32 0
; CHECK: %[[#typeInt64:]] = OpTypeInt 64 0
; CHECK: %[[#Num3:]] = OpConstant %[[#typeInt32]] 3
; CHECK: %[[#Array3x64:]] = OpTypeArray %[[#typeInt64:]] %[[#Num3]]
; CHECK: %[[#TypeNDRangeStruct:]] = OpTypeStruct %[[#typeInt32]] %[[#Array3x64]] %[[#Array3x64]] %[[#Array3x64]]

; CHECK-LABEL: -- Begin function test_ndrange_2D3D
; CHECK: %[[#res1:]] = OpBuildNDRange %[[#TypeNDRangeStruct]] %[[#r11:]] %[[#r12:]] %[[#r12:]]

; CHECK: %[[#GS:]] = OpLoad %[[#Array3x64]] %[[#GSName]]
; CHECK: %[[#LS:]] = OpLoad %8 %[[#LSName]]
; CHECK: %[[#GO:]] = OpLoad %8 %[[#GOName]]
; CHECK: %[[#res2:]] = OpBuildNDRange %[[#TypeNDRangeStruct]] %[[#GS]] %[[#LS]] %[[#GO]]

; CHECK-NOT:    OpStore %[[#ret1:]] %[[#res1]]
; CHECK-NOT:    OpStore %[[#ret2:]] %[[#res2]]

entry:
  call spir_func %struct.ndrange_t @_Z10ndrange_1Dm(i64 noundef 1)
  call spir_func %struct.ndrange_t  @_Z10ndrange_3DPKmS0_S0_(ptr addrspace(2) align 4 @__const.GO_3Dc, ptr addrspace(2) align 4 @__const.GS_3Dc, ptr addrspace(2) align 4 @__const.LS_3Dc)
  ret void
}

declare spir_func %struct.ndrange_t  @_Z10ndrange_1Dm(i64 noundef) local_unnamed_addr #2
declare spir_func %struct.ndrange_t  @_Z10ndrange_3DPKmS0_S0_(ptr noundef) local_unnamed_addr #1
