; Adapted from Khronos Translator test suite: test/CodeGen/SPIRV/extensions/SPV_INTEL_split_barrier/

;; kernel void test(global uint* dst)
;; {
;;    __spirv_ControlBarrierArriveINTEL(2, 2, 260);  // local
;;    __spirv_ControlBarrierWaitINTEL(2, 2, 258);    // local
;;    __spirv_ControlBarrierArriveINTEL(2, 2, 516);  // global
;;    __spirv_ControlBarrierWaitINTEL(2, 2, 514);    // global
;;    __spirv_ControlBarrierArriveINTEL(2, 2, 2052); // image
;;    __spirv_ControlBarrierWaitINTEL(2, 2, 2050);   // image
;;
;;    __spirv_ControlBarrierArriveINTEL(2, 2, 772);  // local + global
;;    __spirv_ControlBarrierWaitINTEL(2, 2, 770);    // local + global
;;    __spirv_ControlBarrierArriveINTEL(2, 2, 2820); // local + global + image
;;    __spirv_ControlBarrierWaitINTEL(2, 2, 2818);   // local + global + image
;;
;;    __spirv_ControlBarrierArriveINTEL(2, 4, 260);  // local, work_item
;;    __spirv_ControlBarrierWaitINTEL(2, 4, 258);    // local, work_item
;;    __spirv_ControlBarrierArriveINTEL(2, 2, 260);  // local, work_group
;;    __spirv_ControlBarrierWaitINTEL(2, 2, 258);    // local, work_group
;;    __spirv_ControlBarrierArriveINTEL(2, 1, 260);  // local, device
;;    __spirv_ControlBarrierWaitINTEL(2, 1, 258);    // local, device
;;    __spirv_ControlBarrierArriveINTEL(2, 0, 260);  // local, all_svm_devices
;;    __spirv_ControlBarrierWaitINTEL(2, 0, 258);    // local, all_svm_devices
;;    __spirv_ControlBarrierArriveINTEL(2, 3, 260);  // local, subgroup
;;    __spirv_ControlBarrierWaitINTEL(2, 3, 258);    // local, subgroup
;;}

; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; CHECK-ERROR: LLVM ERROR: __spirv_ControlBarrierArriveINTEL: the builtin requires the following SPIR-V extension: SPV_INTEL_split_barrier

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_split_barrier %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_split_barrier %s -o - -filetype=obj | spirv-val %}

; CHECK: Capability SplitBarrierINTEL
; CHECK: Extension "SPV_INTEL_split_barrier"
; CHECK: %[[#UINT:]] = OpTypeInt 32 0
;
; Scopes:
; CHECK-DAG: %[[#SCOPE_WORK_GROUP:]] = OpConstant %[[#UINT]] 2
; CHECK-DAG: %[[#SCOPE_INVOCATION:]] = OpConstant %[[#UINT]] 4
; CHECK-DAG: %[[#SCOPE_DEVICE:]] = OpConstant %[[#UINT]] 1
; CHECK-DAG: %[[#SCOPE_CROSS_DEVICE:]] = OpConstant %[[#UINT]] 0
; CHECK-DAG: %[[#SCOPE_SUBGROUP:]] = OpConstant %[[#UINT]] 3
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
; 0x2 Acquire + 0x800 ImageMemory
; CHECK-DAG: %[[#ACQUIRE_IMAGE:]] = OpConstant %[[#UINT]] 2050
; 0x4 Acquire + 0x800 ImageMemory
; CHECK-DAG: %[[#RELEASE_IMAGE:]] = OpConstant %[[#UINT]] 2052
; 0x2 Acquire + 0x100 WorkgroupMemory + 0x200 CrossWorkgroupMemory
; CHECK-DAG: %[[#ACQUIRE_LOCAL_GLOBAL:]] = OpConstant %[[#UINT]] 770
; 0x4 Release + 0x100 WorkgroupMemory + 0x200 CrossWorkgroupMemory
; CHECK-DAG: %[[#RELEASE_LOCAL_GLOBAL:]] = OpConstant %[[#UINT]] 772
; 0x2 Acquire + 0x100 WorkgroupMemory + 0x200 CrossWorkgroupMemory + 0x800 ImageMemory
; CHECK-DAG: %[[#ACQUIRE_LOCAL_GLOBAL_IMAGE:]] = OpConstant %[[#UINT]] 2818
; 0x4 Release + 0x100 WorkgroupMemory + 0x200 CrossWorkgroupMemory + 0x800 ImageMemory
; CHECK-DAG: %[[#RELEASE_LOCAL_GLOBAL_IMAGE:]] = OpConstant %[[#UINT]] 2820
;
; CHECK: OpControlBarrierArriveINTEL %[[#SCOPE_WORK_GROUP]] %[[#SCOPE_WORK_GROUP]] %[[#RELEASE_LOCAL]]
; CHECK: OpControlBarrierWaitINTEL %[[#SCOPE_WORK_GROUP]] %[[#SCOPE_WORK_GROUP]] %[[#ACQUIRE_LOCAL]]
; CHECK: OpControlBarrierArriveINTEL %[[#SCOPE_WORK_GROUP]] %[[#SCOPE_WORK_GROUP]] %[[#RELEASE_GLOBAL]]
; CHECK: OpControlBarrierWaitINTEL %[[#SCOPE_WORK_GROUP]] %[[#SCOPE_WORK_GROUP]] %[[#ACQUIRE_GLOBAL]]
; CHECK: OpControlBarrierArriveINTEL %[[#SCOPE_WORK_GROUP]] %[[#SCOPE_WORK_GROUP]] %[[#RELEASE_IMAGE]]
; CHECK: OpControlBarrierWaitINTEL %[[#SCOPE_WORK_GROUP]] %[[#SCOPE_WORK_GROUP]] %[[#ACQUIRE_IMAGE]]
;
; CHECK: OpControlBarrierArriveINTEL %[[#SCOPE_WORK_GROUP]] %[[#SCOPE_WORK_GROUP]] %[[#RELEASE_LOCAL_GLOBAL]]
; CHECK: OpControlBarrierWaitINTEL %[[#SCOPE_WORK_GROUP]] %[[#SCOPE_WORK_GROUP]] %[[#ACQUIRE_LOCAL_GLOBAL]]
; CHECK: OpControlBarrierArriveINTEL %[[#SCOPE_WORK_GROUP]] %[[#SCOPE_WORK_GROUP]] %[[#RELEASE_LOCAL_GLOBAL_IMAGE]]
; CHECK: OpControlBarrierWaitINTEL %[[#SCOPE_WORK_GROUP]] %[[#SCOPE_WORK_GROUP]] %[[#ACQUIRE_LOCAL_GLOBAL_IMAGE]]
;
; CHECK: OpControlBarrierArriveINTEL %[[#SCOPE_WORK_GROUP]] %[[#SCOPE_INVOCATION]] %[[#RELEASE_LOCAL]]
; CHECK: OpControlBarrierWaitINTEL %[[#SCOPE_WORK_GROUP]] %[[#SCOPE_INVOCATION]] %[[#ACQUIRE_LOCAL]]
; CHECK: OpControlBarrierArriveINTEL %[[#SCOPE_WORK_GROUP]] %[[#SCOPE_WORK_GROUP]] %[[#RELEASE_LOCAL]]
; CHECK: OpControlBarrierWaitINTEL %[[#SCOPE_WORK_GROUP]] %[[#SCOPE_WORK_GROUP]] %[[#ACQUIRE_LOCAL]]
; CHECK: OpControlBarrierArriveINTEL %[[#SCOPE_WORK_GROUP]] %[[#SCOPE_DEVICE]] %[[#RELEASE_LOCAL]]
; CHECK: OpControlBarrierWaitINTEL %[[#SCOPE_WORK_GROUP]] %[[#SCOPE_DEVICE]] %[[#ACQUIRE_LOCAL]]
; CHECK: OpControlBarrierArriveINTEL %[[#SCOPE_WORK_GROUP]] %[[#SCOPE_CROSS_DEVICE]] %[[#RELEASE_LOCAL]]
; CHECK: OpControlBarrierWaitINTEL %[[#SCOPE_WORK_GROUP]] %[[#SCOPE_CROSS_DEVICE]] %[[#ACQUIRE_LOCAL]]
; CHECK: OpControlBarrierArriveINTEL %[[#SCOPE_WORK_GROUP]] %[[#SCOPE_SUBGROUP]] %[[#RELEASE_LOCAL]]
; CHECK: OpControlBarrierWaitINTEL %[[#SCOPE_WORK_GROUP]] %[[#SCOPE_SUBGROUP]] %[[#ACQUIRE_LOCAL]]

; Function Attrs: convergent norecurse nounwind
define dso_local spir_kernel void @test(ptr addrspace(1) nocapture noundef readnone align 4 %0) local_unnamed_addr #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !5 !kernel_arg_type !6 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
  tail call spir_func void @_Z33__spirv_ControlBarrierArriveINTELiii(i32 noundef 2, i32 noundef 2, i32 noundef 260) #2
  tail call spir_func void @_Z31__spirv_ControlBarrierWaitINTELiii(i32 noundef 2, i32 noundef 2, i32 noundef 258) #2
  tail call spir_func void @_Z33__spirv_ControlBarrierArriveINTELiii(i32 noundef 2, i32 noundef 2, i32 noundef 516) #2
  tail call spir_func void @_Z31__spirv_ControlBarrierWaitINTELiii(i32 noundef 2, i32 noundef 2, i32 noundef 514) #2
  tail call spir_func void @_Z33__spirv_ControlBarrierArriveINTELiii(i32 noundef 2, i32 noundef 2, i32 noundef 2052) #2
  tail call spir_func void @_Z31__spirv_ControlBarrierWaitINTELiii(i32 noundef 2, i32 noundef 2, i32 noundef 2050) #2
  tail call spir_func void @_Z33__spirv_ControlBarrierArriveINTELiii(i32 noundef 2, i32 noundef 2, i32 noundef 772) #2
  tail call spir_func void @_Z31__spirv_ControlBarrierWaitINTELiii(i32 noundef 2, i32 noundef 2, i32 noundef 770) #2
  tail call spir_func void @_Z33__spirv_ControlBarrierArriveINTELiii(i32 noundef 2, i32 noundef 2, i32 noundef 2820) #2
  tail call spir_func void @_Z31__spirv_ControlBarrierWaitINTELiii(i32 noundef 2, i32 noundef 2, i32 noundef 2818) #2
  tail call spir_func void @_Z33__spirv_ControlBarrierArriveINTELiii(i32 noundef 2, i32 noundef 4, i32 noundef 260) #2
  tail call spir_func void @_Z31__spirv_ControlBarrierWaitINTELiii(i32 noundef 2, i32 noundef 4, i32 noundef 258) #2
  tail call spir_func void @_Z33__spirv_ControlBarrierArriveINTELiii(i32 noundef 2, i32 noundef 2, i32 noundef 260) #2
  tail call spir_func void @_Z31__spirv_ControlBarrierWaitINTELiii(i32 noundef 2, i32 noundef 2, i32 noundef 258) #2
  tail call spir_func void @_Z33__spirv_ControlBarrierArriveINTELiii(i32 noundef 2, i32 noundef 1, i32 noundef 260) #2
  tail call spir_func void @_Z31__spirv_ControlBarrierWaitINTELiii(i32 noundef 2, i32 noundef 1, i32 noundef 258) #2
  tail call spir_func void @_Z33__spirv_ControlBarrierArriveINTELiii(i32 noundef 2, i32 noundef 0, i32 noundef 260) #2
  tail call spir_func void @_Z31__spirv_ControlBarrierWaitINTELiii(i32 noundef 2, i32 noundef 0, i32 noundef 258) #2
  tail call spir_func void @_Z33__spirv_ControlBarrierArriveINTELiii(i32 noundef 2, i32 noundef 3, i32 noundef 260) #2
  tail call spir_func void @_Z31__spirv_ControlBarrierWaitINTELiii(i32 noundef 2, i32 noundef 3, i32 noundef 258) #2
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func void @_Z33__spirv_ControlBarrierArriveINTELiii(i32 noundef, i32 noundef, i32 noundef) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func void @_Z31__spirv_ControlBarrierWaitINTELiii(i32 noundef, i32 noundef, i32 noundef) local_unnamed_addr #1

attributes #0 = { convergent norecurse nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size"="false" }
attributes #1 = { convergent "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent nounwind }

!llvm.module.flags = !{!0, !1}
!opencl.ocl.version = !{!2}
!opencl.spir.version = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 2, i32 0}
!3 = !{!"clang version 15.0.0 (https://github.com/llvm/llvm-project 861386dbd6ff0d91636b7c674c2abb2eccd9d3f2)"}
!4 = !{i32 1}
!5 = !{!"none"}
!6 = !{!"uint*"}
!7 = !{!""}
