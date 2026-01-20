; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-vulkan-unknown %s -o - | FileCheck %s

%struct.S = type <{ float }>

; CHECK-NOT: OpCapability Linkage

; CHECK-DAG: %[[#F32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#UINT:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#S_S:]] = OpTypeStruct %[[#F32]]

; CHECK-DAG: %[[#PTR_PCS_F:]] = OpTypePointer PushConstant %[[#F32]]
; CHECK-DAG: %[[#PTR_PCS_S:]] = OpTypePointer PushConstant %[[#S_S]]


; CHECK-DAG: OpMemberDecorate %[[#S_S]] 0 Offset 0
; CHECK-DAG: OpDecorate %[[#S_S]] Block


@pcs = external hidden addrspace(13) externally_initialized global %struct.S, align 1
; CHECK: %[[#PCS:]] = OpVariable %[[#PTR_PCS_S]] PushConstant

define void @main() #1 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %1 = alloca float, align 4
  %2 = load float, ptr addrspace(13) @pcs, align 1
  store float %2, ptr %1
  ret void
}

declare token @llvm.experimental.convergence.entry() #2

attributes #1 = { convergent noinline norecurse optnone "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
attributes #2 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }
