; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-vulkan-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-vulkan-unknown %s -o - -filetype=obj | spirv-val %}

; This file generated from the following command:
; clang -cc1 -triple spirv-vulkan-compute -x hlsl -emit-llvm -finclude-default-header -o - - <<EOF
; [numthreads(1, 1, 1)]
; void main() {
;   int idx = WaveGetLaneIndex();
; }
; EOF

; CHECK-DAG:                         OpCapability Shader
; CHECK-DAG:                         OpCapability GroupNonUniform
; CHECK-DAG:                         OpDecorate %[[#var:]] BuiltIn SubgroupLocalInvocationId
; CHECK-DAG:            %[[#int:]] = OpTypeInt 32 0
; CHECK-DAG:           %[[#ptri:]] = OpTypePointer Input %[[#int]]
; CHECK-DAG:           %[[#ptrf:]] = OpTypePointer Function %[[#int]]
; CHECK-DAG:             %[[#var]] = OpVariable %[[#ptri]] Input

; CHECK-NOT:                         OpDecorate %[[#var]] LinkageAttributes


; ModuleID = '-'
source_filename = "-"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spirv-unknown-vulkan-compute"

; Function Attrs: convergent noinline norecurse nounwind optnone
define internal spir_func void @main() #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %idx = alloca i32, align 4
; CHECK:     %[[#idx:]] = OpVariable %[[#ptrf]] Function

  %1 = call i32 @__hlsl_wave_get_lane_index() [ "convergencectrl"(token %0) ]
; CHECK:    %[[#tmp:]] = OpLoad %[[#int]] %[[#var]]

  store i32 %1, ptr %idx, align 4
; CHECK:                 OpStore %[[#idx]] %[[#tmp]]

  ret void
}

; Function Attrs: norecurse
define void @main.1() #1 {
entry:
  call void @main()
  ret void
}

; Function Attrs: convergent
declare i32 @__hlsl_wave_get_lane_index() #2

; Function Attrs: convergent nocallback nofree nosync nounwind willreturn memory(none)
declare token @llvm.experimental.convergence.entry() #3

attributes #0 = { convergent noinline norecurse nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { norecurse "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent }
attributes #3 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 4, !"dx.disable_optimizations", i32 1}
!2 = !{!"clang version 19.0.0git (/usr/local/google/home/nathangauer/projects/llvm-project/clang bc6fd04b73a195981ee77823cf1382d04ab96c44)"}

