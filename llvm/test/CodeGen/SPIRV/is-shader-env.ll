; RUN: llc -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; Regression test for https://github.com/llvm/llvm-project/issues/171898
; When triple is spirv-unknown-unknown and a non-entry-point function using
; wide vectors (e.g. <8 x i32>) appears before the entry point with
; hlsl.shader attribute, the environment must be resolved early enough that
; legalization uses the correct vector size limits.

; CHECK-DAG: OpCapability Shader
; CHECK-DAG: OpEntryPoint GLCompute %[[#entry:]] "main"
; CHECK-NOT: OpTypeVector %{{.*}} 8

@GVec4 = internal addrspace(10) global <4 x double> zeroinitializer
@Lows = internal addrspace(10) global <4 x i32> zeroinitializer
@Highs = internal addrspace(10) global <4 x i32> zeroinitializer

define internal void @test_split() {
entry:
  %0 = load <8 x i32>, ptr addrspace(10) @GVec4, align 32
  %1 = shufflevector <8 x i32> %0, <8 x i32> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %2 = shufflevector <8 x i32> %0, <8 x i32> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  store <4 x i32> %1, ptr addrspace(10) @Lows, align 16
  store <4 x i32> %2, ptr addrspace(10) @Highs, align 16
  ret void
}

define void @main() local_unnamed_addr #0 {
entry:
  call void @test_split()
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
