; RUN: llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; Index 2^31 - 1 should be a 32-bit constant, and index 2^31 a 64-bit one
; that makes the module declare Int64.

; CHECK-DAG: OpCapability Int64
; CHECK-DAG: %[[#uint:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#ulong:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#max32:]] = OpConstant %[[#uint]] 2147483647{{$}}
; CHECK-DAG: %[[#min64:]] = OpConstant %[[#ulong]] 2147483648{{$}}

@g_big = internal addrspace(10) global [2147483649 x float] zeroinitializer
@out = internal addrspace(10) global float zeroinitializer

define void @main() #0 {
entry:
; CHECK: OpInBoundsAccessChain %[[#]] %[[#]] %[[#max32]]{{$}}
  %p0 = getelementptr inbounds nuw i8, ptr addrspace(10) @g_big, i64 8589934588
  %v0 = load float, ptr addrspace(10) %p0
; CHECK: OpInBoundsAccessChain %[[#]] %[[#]] %[[#min64]]{{$}}
  %p1 = getelementptr inbounds nuw i8, ptr addrspace(10) @g_big, i64 8589934592
  %v1 = load float, ptr addrspace(10) %p1
  %sum = fadd float %v0, %v1
  store float %sum, ptr addrspace(10) @out
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
