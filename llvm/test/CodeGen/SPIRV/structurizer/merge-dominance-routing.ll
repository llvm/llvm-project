; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}
; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | FileCheck %s

; Test that the structurizer correctly handles merge dominance when routing
; blocks create alternate paths to merge points. This pattern occurs when
; SPIRVMergeRegionExitTargets creates switch-based routing and the structurizer
; creates conditional branches where a "thread" block (skipping a convergent
; op) provides an alternate path to the merge, breaking header dominance.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv-unknown-vulkan-compute"

; Verify the output has proper nested selection constructs and valid structure.
; The key check: selection merges are properly nested and all branches stay
; within their constructs.
; CHECK:       %[[#entry:]] = OpLabel
; CHECK:                      OpSelectionMerge %[[#merge:]] None
; CHECK:                      OpBranchConditional %[[#]] %[[#wave1:]] %[[#merge]]
; CHECK:    %[[#wave1]] = OpLabel
; CHECK:                      OpSelectionMerge %[[#merge2:]] None
; CHECK:                      OpBranchConditional
; CHECK:   %[[#merge2]] = OpLabel
; CHECK:                      OpBranchConditional %[[#]] %[[#]] %[[#merge]]
; CHECK:    %[[#merge]] = OpLabel
; CHECK:                      OpReturn

define internal spir_func void @main() #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %tid = call i32 @__hlsl_wave_get_lane_index() [ "convergencectrl"(token %0) ]
  %cond1 = icmp ult i32 %tid, 16
  br i1 %cond1, label %wave1, label %thread1

wave1:
  %r1 = call double @llvm.spv.wave.prefix.product.f64(double 1.0) [ "convergencectrl"(token %0) ]
  %cond2 = icmp ult i32 %tid, 8
  br i1 %cond2, label %wave2, label %thread2

wave2:
  %r2 = call double @llvm.spv.wave.prefix.product.f64(double 2.0) [ "convergencectrl"(token %0) ]
  br label %merge

thread2:
  br label %merge

thread1:
  br label %merge

merge:
  %result = phi double [ %r1, %thread2 ], [ %r2, %wave2 ], [ 0.0, %thread1 ]
  ret void
}

declare token @llvm.experimental.convergence.entry() #1
declare i32 @__hlsl_wave_get_lane_index() #2
declare double @llvm.spv.wave.prefix.product.f64(double) #2

attributes #0 = { convergent noinline norecurse nounwind optnone "hlsl.numthreads"="32,1,1" "hlsl.shader"="compute" }
attributes #1 = { convergent nocallback nofree nosync nounwind willreturn }
attributes #2 = { convergent nounwind }
