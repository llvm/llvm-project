; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}
; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | FileCheck %s

; Test that the structurizer correctly handles switch case fall-through ordering.
; When SPIRVMergeRegionExitTargets creates switch-based routing with multiple
; exit targets, and one case construct branches to another (fall-through), the
; cases must be ordered so the falling-through case immediately precedes its
; target in the OpSwitch target list.
;
; This pattern occurs with 4+ convergent operations in nested conditionals,
; where the routing switch has cases that form a chain.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv-unknown-vulkan-compute"

; CHECK: OpSelectionMerge
; CHECK: OpBranchConditional

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
  %cond3 = icmp ult i32 %tid, 4
  br i1 %cond3, label %wave3, label %thread3

wave3:
  %r3 = call double @llvm.spv.wave.prefix.product.f64(double 3.0) [ "convergencectrl"(token %0) ]
  %cond4 = icmp ult i32 %tid, 2
  br i1 %cond4, label %wave4, label %thread4

wave4:
  %r4 = call double @llvm.spv.wave.prefix.product.f64(double 4.0) [ "convergencectrl"(token %0) ]
  br label %merge

thread4:
  br label %merge

thread3:
  br label %merge

thread2:
  br label %merge

thread1:
  br label %merge

merge:
  %result = phi double [ %r4, %wave4 ], [ 0.0, %thread4 ], [ 0.0, %thread3 ], [ 0.0, %thread2 ], [ 0.0, %thread1 ]
  ret void
}

declare token @llvm.experimental.convergence.entry() #1
declare i32 @__hlsl_wave_get_lane_index() #2
declare double @llvm.spv.wave.prefix.product.f64(double) #2

attributes #0 = { convergent noinline norecurse nounwind optnone "hlsl.numthreads"="32,1,1" "hlsl.shader"="compute" }
attributes #1 = { convergent nocallback nofree nosync nounwind willreturn }
attributes #2 = { convergent nounwind }
