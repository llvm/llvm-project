; Adapted from Khronos Translator test suite: test/CodeGen/SPIRV/extensions/SPV_INTEL_split_barrier/

;; kernel void test(global uint* dst)
;; {
;;     intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);
;;     intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);
;;     intel_work_group_barrier_arrive(CLK_GLOBAL_MEM_FENCE);
;;     intel_work_group_barrier_wait(CLK_GLOBAL_MEM_FENCE);
;;
;;     intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
;;     intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
;;}

; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; CHECK-ERROR: LLVM ERROR: intel_work_group_barrier_arrive: the builtin requires the following SPIR-V extension: SPV_INTEL_split_barrier

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_split_barrier %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_split_barrier %s -o - -filetype=obj | spirv-val %}

; CHECK: Capability SplitBarrierINTEL
; CHECK: Extension "SPV_INTEL_split_barrier"
; CHECK: %[[#UINT:]] = OpTypeInt 32 0
;
; Scopes:
; CHECK-DAG: %[[#SCOPE_WORK_GROUP:]] = OpConstant %[[#UINT]] 2{{$}}
;
; Memory Semantics:
; 0x2 Acquire + 0x100 WorkgroupMemory
; CHECK-DAG: %[[#ACQUIRE_LOCAL:]] = OpConstant %[[#UINT]] 258
; 0x4 Release + 0x100 WorkgroupMemory
; CHECK-DAG: %[[#RELEASE_LOCAL:]] = OpConstant %[[#UINT]] 260
; 0x2 Acquire + 0x200 CrossWorkgroupMemory
; CHECK-DAG: %[[#ACQUIRE_GLOBAL:]] = OpConstant %[[#UINT]] 514
; 0x4 Release + 0x200 CrossWorkgroupMemory
; CHECK-DAG: %[[#RELEASE_GLOBAL:]] = OpConstant %[[#UINT]] 516
; 0x2 Acquire + 0x100 WorkgroupMemory + 0x200 CrossWorkgroupMemory
; CHECK-DAG: %[[#ACQUIRE_LOCAL_GLOBAL:]] = OpConstant %[[#UINT]] 770
; 0x4 Release + 0x100 WorkgroupMemory + 0x200 CrossWorkgroupMemory
; CHECK-DAG: %[[#RELEASE_LOCAL_GLOBAL:]] = OpConstant %[[#UINT]] 772
;
; CHECK: OpControlBarrierArriveINTEL %[[#SCOPE_WORK_GROUP]] %[[#SCOPE_WORK_GROUP]] %[[#RELEASE_LOCAL]]
; CHECK: OpControlBarrierWaitINTEL %[[#SCOPE_WORK_GROUP]] %[[#SCOPE_WORK_GROUP]] %[[#ACQUIRE_LOCAL]]
; CHECK: OpControlBarrierArriveINTEL %[[#SCOPE_WORK_GROUP]] %[[#SCOPE_WORK_GROUP]] %[[#RELEASE_GLOBAL]]
; CHECK: OpControlBarrierWaitINTEL %[[#SCOPE_WORK_GROUP]] %[[#SCOPE_WORK_GROUP]] %[[#ACQUIRE_GLOBAL]]
;
; CHECK: OpControlBarrierArriveINTEL %[[#SCOPE_WORK_GROUP]] %[[#SCOPE_WORK_GROUP]] %[[#RELEASE_LOCAL_GLOBAL]]
; CHECK: OpControlBarrierWaitINTEL %[[#SCOPE_WORK_GROUP]] %[[#SCOPE_WORK_GROUP]] %[[#ACQUIRE_LOCAL_GLOBAL]]

; Function Attrs: convergent norecurse nounwind
define dso_local spir_kernel void @test(ptr addrspace(1) nocapture noundef readnone align 4 %0) local_unnamed_addr #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !5 !kernel_arg_type !6 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
  tail call spir_func void @_Z31intel_work_group_barrier_arrivej(i32 noundef 1) #2
  tail call spir_func void @_Z29intel_work_group_barrier_waitj(i32 noundef 1) #2
  tail call spir_func void @_Z31intel_work_group_barrier_arrivej(i32 noundef 2) #2
  tail call spir_func void @_Z29intel_work_group_barrier_waitj(i32 noundef 2) #2
  tail call spir_func void @_Z31intel_work_group_barrier_arrivej(i32 noundef 3) #2
  tail call spir_func void @_Z29intel_work_group_barrier_waitj(i32 noundef 3) #2
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func void @_Z31intel_work_group_barrier_arrivej(i32 noundef) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func void @_Z29intel_work_group_barrier_waitj(i32 noundef) local_unnamed_addr #1

attributes #0 = { convergent norecurse nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" }
attributes #1 = { convergent "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent nounwind }

!llvm.module.flags = !{!0, !1}
!opencl.ocl.version = !{!2}
!opencl.spir.version = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 1, i32 2}
!3 = !{!"clang version 15.0.0 (https://github.com/llvm/llvm-project 861386dbd6ff0d91636b7c674c2abb2eccd9d3f2)"}
!4 = !{i32 1}
!5 = !{!"none"}
!6 = !{!"uint*"}
!7 = !{!""}
