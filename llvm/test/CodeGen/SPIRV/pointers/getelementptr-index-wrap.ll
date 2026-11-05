; RUN: llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; With an i32 index type, each constant GEP should select element 1 of @a, and
; each dynamic one should index the access chain with a 32-bit value.

; CHECK-DAG: %[[#uint:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#uint_1:]] = OpConstant %[[#uint]] 1{{$}}
; CHECK-DAG: %[[#uint_4:]] = OpConstant %[[#uint]] 4{{$}}

@a = internal addrspace(10) global [4 x i32] [i32 10, i32 20, i32 30, i32 40]
@idx = internal addrspace(10) global i64 0
@out = internal addrspace(10) global i32 0

define void @main() #0 {
entry:
; The scaled offset 1073741825 * 4 wraps to 4.
; CHECK: OpAccessChain %[[#]] %[[#]] %[[#uint_1]]{{$}}
  %p0 = getelementptr [4 x i8], ptr addrspace(10) @a, i32 1073741825
  %v0 = load i32, ptr addrspace(10) %p0
; The byte offset 4294967300 truncates to 4.
; CHECK: OpAccessChain %[[#]] %[[#]] %[[#uint_1]]{{$}}
  %p1 = getelementptr i8, ptr addrspace(10) @a, i64 4294967300
  %v1 = load i32, ptr addrspace(10) %p1
; The array index 4294967297 truncates to 1.
; CHECK: OpAccessChain %[[#]] %[[#]] %[[#uint_1]]{{$}}
  %p2 = getelementptr [4 x i32], ptr addrspace(10) @a, i32 0, i64 4294967297
  %v2 = load i32, ptr addrspace(10) %p2
  %i = load i64, ptr addrspace(10) @idx
; CHECK: %[[#idx:]] = OpUConvert %[[#uint]] %[[#]]
; CHECK: OpAccessChain %[[#]] %[[#]] %[[#idx]]{{$}}
  %p3 = getelementptr [4 x i32], ptr addrspace(10) @a, i32 0, i64 %i
  %v3 = load i32, ptr addrspace(10) %p3
; CHECK: %[[#off:]] = OpUConvert %[[#uint]] %[[#]]
; CHECK: %[[#div:]] = OpUDiv %[[#uint]] %[[#off]] %[[#uint_4]]
; CHECK: OpAccessChain %[[#]] %[[#]] %[[#div]]{{$}}
  %p4 = getelementptr i8, ptr addrspace(10) @a, i64 %i
  %v4 = load i32, ptr addrspace(10) %p4
  %s0 = add i32 %v0, %v1
  %s1 = add i32 %v2, %v3
  %s2 = add i32 %s0, %s1
  %s3 = add i32 %s2, %v4
  store i32 %s3, ptr addrspace(10) @out
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
