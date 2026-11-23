; RUN: opt -mtriple=spirv-unknown-vulkan1.3-compute -passes=spirv-finalize-shader-linkage -S %s | FileCheck %s --check-prefix=OPT
; RUN: llc  -mtriple=spirv-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -mtriple=spirv-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val %}

; The pass erases the dead helper and keeps the entry point.
; OPT-NOT: @dead_matrix_helper
; OPT: define void @main()

; CHECK-DAG: OpEntryPoint GLCompute %[[#entry:]] "main"
; The dead helper (and its illegal wide-vector parameter) must not be emitted.
; CHECK-NOT: OpFunctionParameter
; CHECK-NOT: dead_matrix_helper

define hidden spir_func <6 x float> @dead_matrix_helper(<6 x float> %m) #1 {
  %r = fadd <6 x float> %m, %m
  ret <6 x float> %r
}

define void @main() #0 {
entry:
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
attributes #1 = { alwaysinline }
