; RUN: llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - | FileCheck %s --implicit-check-not="Int64"
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; Constant byte offsets into a struct, a nested struct, a vector and an array
; should become 32-bit access chain indices and leave Int64 undeclared.

; CHECK-DAG: %[[#uint:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#uint_1:]] = OpConstant %[[#uint]] 1{{$}}
; CHECK-DAG: %[[#uint_2:]] = OpConstant %[[#uint]] 2{{$}}
; CHECK-DAG: %[[#uint_3:]] = OpConstant %[[#uint]] 3{{$}}

%S = type { float, float }
%Outer = type { float, %S }

@g_s = internal addrspace(10) global %S zeroinitializer
@g_outer = internal addrspace(10) global %Outer zeroinitializer
@g_vec = internal addrspace(10) global <4 x float> zeroinitializer
@g_arr = internal addrspace(10) global [4 x float] zeroinitializer
@out = internal addrspace(10) global float zeroinitializer

define void @main() #0 {
entry:
; CHECK: OpInBoundsAccessChain %[[#]] %[[#]] %[[#uint_1]]{{$}}
  %p0 = getelementptr inbounds nuw i8, ptr addrspace(10) @g_s, i64 4
  %v0 = load float, ptr addrspace(10) %p0
; CHECK: OpInBoundsAccessChain %[[#]] %[[#]] %[[#uint_1]] %[[#uint_1]]{{$}}
  %p1 = getelementptr inbounds nuw i8, ptr addrspace(10) @g_outer, i64 8
  %v1 = load float, ptr addrspace(10) %p1
; CHECK: OpInBoundsAccessChain %[[#]] %[[#]] %[[#uint_2]]{{$}}
  %p2 = getelementptr inbounds nuw i8, ptr addrspace(10) @g_vec, i64 8
  %v2 = load float, ptr addrspace(10) %p2
; CHECK: OpInBoundsAccessChain %[[#]] %[[#]] %[[#uint_3]]{{$}}
  %p3 = getelementptr inbounds nuw i8, ptr addrspace(10) @g_arr, i64 12
  %v3 = load float, ptr addrspace(10) %p3
  %a = fadd float %v0, %v1
  %b = fadd float %v2, %v3
  %c = fadd float %a, %b
  store float %c, ptr addrspace(10) @out
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
