; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-vulkan-unknown %s -o - | FileCheck %s

%struct.anon = type { i32, float, <3 x float> }

; CHECK-DAG: %[[#F32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#V3F32:]] = OpTypeVector %[[#F32]] 3
; CHECK-DAG: %[[#UINT:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#STRUCT:]] = OpTypeStruct %[[#UINT]] %[[#F32]] %[[#V3F32]]

; CHECK-DAG: OpMemberDecorate %[[#STRUCT]] 0 Offset 0
; CHECK-DAG: OpMemberDecorate %[[#STRUCT]] 1 Offset 4
; CHECK-DAG: OpMemberDecorate %[[#STRUCT]] 2 Offset 8
; CHECK-DAG: OpDecorate %[[#STRUCT]] Block

; CHECK-DAG: %[[#PTR_PCS:]] = OpTypePointer PushConstant %[[#STRUCT]]

@PushConstants = external hidden addrspace(13) externally_initialized global %struct.anon, align 1
; CHECK: %[[#PCS:]] = OpVariable %[[#PTR_PCS]] PushConstant

define void @main() #1 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  ret void
}

declare token @llvm.experimental.convergence.entry() #2

attributes #1 = { convergent noinline norecurse optnone "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
attributes #2 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }
