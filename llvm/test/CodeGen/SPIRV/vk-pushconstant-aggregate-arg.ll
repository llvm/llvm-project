; A function with an aggregate argument that reads a push constant. Removing
; the aggregate types clones the function, so it must run before
; SPIRVPushConstantAccess records the access in SPIRVGlobalRegistry, or the
; record points into the erased original.

; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-vulkan-unknown %s -o - | FileCheck %s
; RUN: llc -O0 -verify-machineinstrs -enable-new-pm -mtriple=spirv-vulkan-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-vulkan-unknown %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK-DAG: %[[#F32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#S:]] = OpTypeStruct %[[#F32]]{{$}}
; CHECK-DAG: %[[#PTR:]] = OpTypePointer PushConstant %[[#S]]
; CHECK-DAG: %[[#PTRF:]] = OpTypePointer PushConstant %[[#F32]]
; CHECK-DAG: %[[#PCS:]] = OpVariable %[[#PTR]] PushConstant
; CHECK: OpFunction
; CHECK: OpFunctionParameter
; CHECK: %[[#AC:]] = OpAccessChain %[[#PTRF]] %[[#PCS]]
; CHECK: OpLoad %[[#F32]] %[[#AC]]

%struct.S = type <{ float }>
%struct.P = type { float, float }

@pcs = external hidden addrspace(13) externally_initialized global %struct.S, align 1

define internal float @helper(%struct.P %p) {
entry:
  %v = load float, ptr addrspace(13) @pcs, align 1
  %x = extractvalue %struct.P %p, 0
  %r = fadd float %v, %x
  ret float %r
}

define void @main() #1 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %1 = alloca float, align 4
  %r = call float @helper(%struct.P { float 1.0, float 2.0 })
  store float %r, ptr %1
  ret void
}

declare token @llvm.experimental.convergence.entry() #2

attributes #1 = { convergent noinline norecurse optnone "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
attributes #2 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }
