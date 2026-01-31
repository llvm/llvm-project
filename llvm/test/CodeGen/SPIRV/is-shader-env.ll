; RUN: llc -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Regression test for https://github.com/llvm/llvm-project/issues/171898
; When triple is spirv-unknown-unknown and a non-entry-point function using
; wide vectors (e.g. <8 x i32>) appears before the entry point with
; hlsl.shader attribute, the environment must be resolved early enough that
; legalization uses the correct vector size limits.

; CHECK-DAG: OpCapability Shader
; CHECK-DAG: OpEntryPoint GLCompute %[[#entry:]] "main"

define <4 x i32> @helper(<4 x i32> %a, <4 x i32> %b) {
entry:
  %result = add <4 x i32> %a, %b
  ret <4 x i32> %result
}

define void @main() #0 {
entry:
  %a = call <4 x i32> @helper(<4 x i32> <i32 1, i32 2, i32 3, i32 4>, <4 x i32> <i32 5, i32 6, i32 7, i32 8>)
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
