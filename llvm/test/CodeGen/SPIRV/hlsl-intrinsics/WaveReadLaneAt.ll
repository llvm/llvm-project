; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32v1.3-vulkan-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-vulkan-unknown %s -o - -filetype=obj | spirv-val %}

; Test lowering to spir-v backend

; CHECK-DAG:   %[[#uint:]] = OpTypeInt 32 0
; CHECK-DAG:   %[[#scope:]] = OpConstant %[[#uint]] 2
; CHECK-DAG:   %[[#f32:]] = OpTypeFloat 32
; CHECK-DAG:   %[[#expr:]] = OpFunctionParameter %[[#f32]]
; CHECK-DAG:   %[[#idx:]] = OpFunctionParameter %[[#uint]]

define spir_func void @test_1(float %expr, i32 %idx) #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
; CHECK:   %[[#ret:]] = OpGroupNonUniformShuffle %[[#f32]] %[[#expr]] %[[#idx]] %[[#scope]]
  %1 = call float @llvm.spv.wave.read.lane.at(float %expr, i32 %idx) [ "convergencectrl"(token %0) ]
  ret void
}

declare i32 @__hlsl_wave_get_lane_index() #1

attributes #0 = { convergent norecurse "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { convergent }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 4, !"dx.disable_optimizations", i32 1}
