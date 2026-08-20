; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; An Interlocked op on a groupshared destination must use the Workgroup scope
; (2), not CrossDevice (which the backend emits as OpConstantNull).

@gs = external hidden addrspace(3) global i32, align 4

; CHECK-DAG: %[[#UINT:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#WORKGROUP:]] = OpConstant %[[#UINT]] 2{{$}}
; CHECK: OpAtomicOr %[[#UINT]] %[[#]] %[[#WORKGROUP]] %[[#]] %[[#]]
; CHECK-NOT: OpConstantNull

define void @main() #0 {
entry:
  %0 = atomicrmw or ptr addrspace(3) @gs, i32 1 syncscope("workgroup") monotonic, align 4
  ret void
}

attributes #0 = { convergent noinline norecurse "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
