; RUN: llc -O0 -mtriple=spirv-unknown-linux %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-vulkan-unknown %s -o - -filetype=obj | spirv-val %}

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spirv-unknown-vulkan-compute"

; CHECK-DAG:   %[[#uint:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#uint_3:]] = OpConstant %[[#uint]] 3
; CHECK-DAG:   %[[#bool:]] = OpTypeBool

define spir_func void @main() #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
; CHECK:   %[[#]] = OpGroupNonUniformElect %[[#bool]] %[[#uint_3]]
  %1 = call i1 @llvm.spv.wave.is.first.lane() [ "convergencectrl"(token %0) ]
  ret void
}

declare i32 @__hlsl_wave_get_lane_index() #1

attributes #0 = { convergent norecurse "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { convergent }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 4, !"dx.disable_optimizations", i32 1}
