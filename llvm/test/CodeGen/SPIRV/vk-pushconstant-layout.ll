; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-vulkan-unknown %s -o - | FileCheck %s
; XFAIL: *
; FIXME(168401): fix the offset of last struct S field.

%struct.T = type { [3 x <2 x float>] }
%struct.S = type <{ float, <3 x float>, %struct.T }>

; CHECK-NOT: OpCapability Linkage

; CHECK-DAG: %[[#PTR_PCS:]] = OpTypePointer PushConstant %[[#S_S:]]

; CHECK-DAG: %[[#F32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#V3F32:]] = OpTypeVector %[[#F32]] 3
; CHECK-DAG: %[[#V2F32:]] = OpTypeVector %[[#F32]] 2
; CHECK-DAG: %[[#UINT:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#UINT_3:]] = OpConstant %[[#UINT]] 3

; CHECK-DAG: %[[#S_S]] = OpTypeStruct %[[#F32]] %[[#V3F32]] %[[#S_T:]]
; CHECK-DAG: %[[#S_T]] = OpTypeStruct %[[#ARR:]]
; CHECK-DAG: %[[#ARR]] = OpTypeArray %[[#V2F32]] %[[#UINT_3]]

; CHECK-DAG: OpMemberDecorate %[[#S_T]] 0 Offset 0
; CHECK-DAG: OpMemberDecorate %[[#S_S]] 0 Offset 0
; CHECK-DAG: OpMemberDecorate %[[#S_S]] 1 Offset 4
; CHECK-DAG: OpMemberDecorate %[[#S_S]] 2 Offset 16
; CHECK-DAG: OpDecorate %[[#S_S]] Block
; CHECK-DAG: OpDecorate %[[#ARR]] ArrayStride 8


@pcs = external hidden addrspace(13) externally_initialized global %struct.S, align 1
; CHECK: %[[#PCS:]] = OpVariable %[[#PTR_PCS]] PushConstant

define void @main() #1 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  ret void
}

declare token @llvm.experimental.convergence.entry() #2

attributes #1 = { convergent noinline norecurse optnone "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
attributes #2 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }
