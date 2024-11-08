; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

;; This test checks that the backend is capable to correctly translate
;; sub_group_barrier built-in function [1] from cl_khr_subgroups extension into
;; corresponding SPIR-V instruction.

;; __kernel void test_barrier_const_flags() {
;;   work_group_barrier(CLK_LOCAL_MEM_FENCE);
;;   work_group_barrier(CLK_GLOBAL_MEM_FENCE);
;;   work_group_barrier(CLK_IMAGE_MEM_FENCE);
;;
;;   work_group_barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
;;   work_group_barrier(CLK_LOCAL_MEM_FENCE | CLK_IMAGE_MEM_FENCE);
;;   work_group_barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE | CLK_IMAGE_MEM_FENCE);
;;
;;   work_group_barrier(CLK_LOCAL_MEM_FENCE, memory_scope_work_item);
;;   work_group_barrier(CLK_LOCAL_MEM_FENCE, memory_scope_work_group);
;;   work_group_barrier(CLK_LOCAL_MEM_FENCE, memory_scope_device);
;;   work_group_barrier(CLK_LOCAL_MEM_FENCE, memory_scope_all_svm_devices);
;;   work_group_barrier(CLK_LOCAL_MEM_FENCE, memory_scope_sub_group);
;;
  ;; barrier should also work (preserved for backward compatibility)
;;   barrier(CLK_GLOBAL_MEM_FENCE);
;; }
;;
;; __kernel void test_barrier_non_const_flags(cl_mem_fence_flags flags, memory_scope scope) {
  ;; FIXME: OpenCL spec doesn't require flags to be compile-time known
  ;; work_group_barrier(flags);
  ;; work_group_barrier(flags, scope);
;; }

; CHECK-SPIRV: OpName %[[#TEST_CONST_FLAGS:]] "test_barrier_const_flags"
; CHECK-SPIRV: %[[#UINT:]] = OpTypeInt 32 0

;; In SPIR-V, barrier is represented as OpControlBarrier [2] and OpenCL
;; cl_mem_fence_flags are represented as part of Memory Semantics [3], which
;; also includes memory order constraints. The backend applies some default
;; memory order for OpControlBarrier and therefore, constants below include a
;; bit more information than original source

;; 0x10 SequentiallyConsistent + 0x100 WorkgroupMemory
; CHECK-SPIRV-DAG: %[[#LOCAL:]] = OpConstant %[[#UINT]] 272
;; 0x10 SequentiallyConsistent + 0x200 CrossWorkgroupMemory
; CHECK-SPIRV-DAG: %[[#GLOBAL:]] = OpConstant %[[#UINT]] 528
;; 0x10 SequentiallyConsistent + 0x800 ImageMemory
; CHECK-SPIRV-DAG: %[[#IMAGE:]] = OpConstant %[[#UINT]] 2064
;; 0x10 SequentiallyConsistent + 0x100 WorkgroupMemory + 0x200 CrossWorkgroupMemory
; CHECK-SPIRV-DAG: %[[#LOCAL_GLOBAL:]] = OpConstant %[[#UINT]] 784
;; 0x10 SequentiallyConsistent + 0x100 WorkgroupMemory + 0x800 ImageMemory
; CHECK-SPIRV-DAG: %[[#LOCAL_IMAGE:]] = OpConstant %[[#UINT]] 2320
;; 0x10 SequentiallyConsistent + 0x100 WorkgroupMemory + 0x200 CrossWorkgroupMemory + 0x800 ImageMemory
; CHECK-SPIRV-DAG: %[[#LOCAL_GLOBAL_IMAGE:]] = OpConstant %[[#UINT]] 2832

;; Scopes [4]:
;; 2 Workgroup
; CHECK-SPIRV-DAG: %[[#SCOPE_WORK_GROUP:]] = OpConstant %[[#UINT]] 2
;; 4 Invocation
; CHECK-SPIRV-DAG: %[[#SCOPE_INVOCATION:]] = OpConstant %[[#UINT]] 4
;; 1 Device
; CHECK-SPIRV-DAG: %[[#SCOPE_DEVICE:]] = OpConstant %[[#UINT]] 1
;; 0 CrossDevice
; CHECK-SPIRV-DAG: %[[#SCOPE_CROSS_DEVICE:]] = OpConstant %[[#UINT]] 0
;; 3 Subgroup
; CHECK-SPIRV-DAG: %[[#SCOPE_SUBGROUP:]] = OpConstant %[[#UINT]] 3

; CHECK-SPIRV: %[[#TEST_CONST_FLAGS]] = OpFunction %[[#]]
; CHECK-SPIRV: OpControlBarrier %[[#SCOPE_WORK_GROUP]] %[[#SCOPE_WORK_GROUP]] %[[#LOCAL]]
; CHECK-SPIRV: OpControlBarrier %[[#SCOPE_WORK_GROUP]] %[[#SCOPE_WORK_GROUP]] %[[#GLOBAL]]
; CHECK-SPIRV: OpControlBarrier %[[#SCOPE_WORK_GROUP]] %[[#SCOPE_WORK_GROUP]] %[[#IMAGE]]
; CHECK-SPIRV: OpControlBarrier %[[#SCOPE_WORK_GROUP]] %[[#SCOPE_WORK_GROUP]] %[[#LOCAL_GLOBAL]]
; CHECK-SPIRV: OpControlBarrier %[[#SCOPE_WORK_GROUP]] %[[#SCOPE_WORK_GROUP]] %[[#LOCAL_IMAGE]]
; CHECK-SPIRV: OpControlBarrier %[[#SCOPE_WORK_GROUP]] %[[#SCOPE_WORK_GROUP]] %[[#LOCAL_GLOBAL_IMAGE]]
; CHECK-SPIRV: OpControlBarrier %[[#SCOPE_WORK_GROUP]] %[[#SCOPE_INVOCATION]] %[[#LOCAL]]
; CHECK-SPIRV: OpControlBarrier %[[#SCOPE_WORK_GROUP]] %[[#SCOPE_WORK_GROUP]] %[[#LOCAL]]
; CHECK-SPIRV: OpControlBarrier %[[#SCOPE_WORK_GROUP]] %[[#SCOPE_DEVICE]] %[[#LOCAL]]
; CHECK-SPIRV: OpControlBarrier %[[#SCOPE_WORK_GROUP]] %[[#SCOPE_CROSS_DEVICE]] %[[#LOCAL]]
; CHECK-SPIRV: OpControlBarrier %[[#SCOPE_WORK_GROUP]] %[[#SCOPE_SUBGROUP]] %[[#LOCAL]]
; CHECK-SPIRV: OpControlBarrier %[[#SCOPE_WORK_GROUP]] %[[#SCOPE_WORK_GROUP]] %[[#GLOBAL]]

define dso_local spir_kernel void @test_barrier_const_flags() local_unnamed_addr {
entry:
  tail call spir_func void @_Z18work_group_barrierj(i32 noundef 1)
  tail call spir_func void @_Z18work_group_barrierj(i32 noundef 2)
  tail call spir_func void @_Z18work_group_barrierj(i32 noundef 4)
  tail call spir_func void @_Z18work_group_barrierj(i32 noundef 3)
  tail call spir_func void @_Z18work_group_barrierj(i32 noundef 5)
  tail call spir_func void @_Z18work_group_barrierj(i32 noundef 7)
  tail call spir_func void @_Z18work_group_barrierj12memory_scope(i32 noundef 1, i32 noundef 0)
  tail call spir_func void @_Z18work_group_barrierj12memory_scope(i32 noundef 1, i32 noundef 1)
  tail call spir_func void @_Z18work_group_barrierj12memory_scope(i32 noundef 1, i32 noundef 2)
  tail call spir_func void @_Z18work_group_barrierj12memory_scope(i32 noundef 1, i32 noundef 3)
  tail call spir_func void @_Z18work_group_barrierj12memory_scope(i32 noundef 1, i32 noundef 4)
  tail call spir_func void @_Z7barrierj(i32 noundef 2)
  ret void
}

declare spir_func void @_Z18work_group_barrierj(i32 noundef) local_unnamed_addr

declare spir_func void @_Z18work_group_barrierj12memory_scope(i32 noundef, i32 noundef) local_unnamed_addr

declare spir_func void @_Z7barrierj(i32 noundef) local_unnamed_addr

define dso_local spir_kernel void @test_barrier_non_const_flags(i32 noundef %flags, i32 noundef %scope) local_unnamed_addr {
entry:
  ret void
}

;; References:
;; [1]: https://www.khronos.org/registry/OpenCL/sdk/2.0/docs/man/xhtml/work_group_barrier.html
;; [2]: https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#OpControlBarrier
;; [3]: https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#_a_id_memory_semantics__id_a_memory_semantics_lt_id_gt
