; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

;; This test checks that the backend is capable to correctly translate
;; atomic_work_item_fence OpenCL C 2.0 built-in function [1] into corresponding
;; SPIR-V instruction [2].

;; __kernel void test_mem_fence_const_flags() {
;;   atomic_work_item_fence(CLK_LOCAL_MEM_FENCE, memory_order_relaxed, memory_scope_work_item);
;;   atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_acquire, memory_scope_work_group);
;;   atomic_work_item_fence(CLK_IMAGE_MEM_FENCE, memory_order_release, memory_scope_device);
;;   atomic_work_item_fence(CLK_LOCAL_MEM_FENCE, memory_order_acq_rel, memory_scope_all_svm_devices);
;;   atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_seq_cst, memory_scope_sub_group);
;;   atomic_work_item_fence(CLK_IMAGE_MEM_FENCE | CLK_LOCAL_MEM_FENCE, memory_order_acquire, memory_scope_sub_group);
;; }

;; __kernel void test_mem_fence_non_const_flags(cl_mem_fence_flags flags, memory_order order, memory_scope scope) {
;;   // FIXME: OpenCL spec doesn't require flags to be compile-time known
;;   // atomic_work_item_fence(flags, order, scope);
;; }

; CHECK-SPIRV:     OpName %[[#TEST_CONST_FLAGS:]] "test_mem_fence_const_flags"
; CHECK-SPIRV:     %[[#UINT:]] = OpTypeInt 32 0

;; 0x0 Relaxed + 0x100 WorkgroupMemory
; CHECK-SPIRV-DAG: %[[#LOCAL_RELAXED:]] = OpConstant %[[#UINT]] 256
;; 0x2 Acquire + 0x200 CrossWorkgroupMemory
; CHECK-SPIRV-DAG: %[[#GLOBAL_ACQUIRE:]] = OpConstant %[[#UINT]] 514
;; 0x4 Release + 0x800 ImageMemory
; CHECK-SPIRV-DAG: %[[#IMAGE_RELEASE:]] = OpConstant %[[#UINT]] 2052
;; 0x8 AcquireRelease + 0x100 WorkgroupMemory
; CHECK-SPIRV-DAG: %[[#LOCAL_ACQ_REL:]] = OpConstant %[[#UINT]] 264
;; 0x10 SequentiallyConsistent + 0x200 CrossWorkgroupMemory
; CHECK-SPIRV-DAG: %[[#GLOBAL_SEQ_CST:]] = OpConstant %[[#UINT]] 528
;; 0x2 Acquire + 0x100 WorkgroupMemory + 0x800 ImageMemory
; CHECK-SPIRV-DAG: %[[#LOCAL_IMAGE_ACQUIRE:]] = OpConstant %[[#UINT]] 2306

;; Scopes [4]:
;; 4 Invocation
; CHECK-SPIRV-DAG: %[[#SCOPE_INVOCATION:]] = OpConstant %[[#UINT]] 4
;; 2 Workgroup
; CHECK-SPIRV-DAG: %[[#SCOPE_WORK_GROUP:]] = OpConstant %[[#UINT]] 2
;; 1 Device
; CHECK-SPIRV-DAG: %[[#SCOPE_DEVICE:]] = OpConstant %[[#UINT]] 1
;; 0 CrossDevice
; CHECK-SPIRV-DAG: %[[#SCOPE_CROSS_DEVICE:]] = OpConstant %[[#UINT]] 0
;; 3 Subgroup
; CHECK-SPIRV-DAG: %[[#SCOPE_SUBGROUP:]] = OpConstant %[[#UINT]] 3

; CHECK-SPIRV: %[[#TEST_CONST_FLAGS]] = OpFunction %[[#]]
; CHECK-SPIRV: OpMemoryBarrier %[[#SCOPE_INVOCATION]] %[[#LOCAL_RELAXED]]
; CHECK-SPIRV: OpMemoryBarrier %[[#SCOPE_WORK_GROUP]] %[[#GLOBAL_ACQUIRE]]
; CHECK-SPIRV: OpMemoryBarrier %[[#SCOPE_DEVICE]] %[[#IMAGE_RELEASE]]
; CHECK-SPIRV: OpMemoryBarrier %[[#SCOPE_CROSS_DEVICE]] %[[#LOCAL_ACQ_REL]]
; CHECK-SPIRV: OpMemoryBarrier %[[#SCOPE_SUBGROUP]] %[[#GLOBAL_SEQ_CST]]
; CHECK-SPIRV: OpMemoryBarrier %[[#SCOPE_SUBGROUP]] %[[#LOCAL_IMAGE_ACQUIRE]]

define dso_local spir_kernel void @test_mem_fence_const_flags() local_unnamed_addr {
entry:
  tail call spir_func void @_Z22atomic_work_item_fencej12memory_order12memory_scope(i32 noundef 1, i32 noundef 0, i32 noundef 0)
  tail call spir_func void @_Z22atomic_work_item_fencej12memory_order12memory_scope(i32 noundef 2, i32 noundef 2, i32 noundef 1)
  tail call spir_func void @_Z22atomic_work_item_fencej12memory_order12memory_scope(i32 noundef 4, i32 noundef 3, i32 noundef 2)
  tail call spir_func void @_Z22atomic_work_item_fencej12memory_order12memory_scope(i32 noundef 1, i32 noundef 4, i32 noundef 3)
  tail call spir_func void @_Z22atomic_work_item_fencej12memory_order12memory_scope(i32 noundef 2, i32 noundef 5, i32 noundef 4)
  tail call spir_func void @_Z22atomic_work_item_fencej12memory_order12memory_scope(i32 noundef 5, i32 noundef 2, i32 noundef 4)
  ret void
}

declare spir_func void @_Z22atomic_work_item_fencej12memory_order12memory_scope(i32 noundef, i32 noundef, i32 noundef) local_unnamed_addr

define dso_local spir_kernel void @test_mem_fence_non_const_flags(i32 noundef %flags, i32 noundef %order, i32 noundef %scope) local_unnamed_addr {
entry:
  ret void
}

;; References:
;; [1]: https://www.khronos.org/registry/OpenCL/sdk/2.0/docs/man/xhtml/atomic_work_item_fence.html
;; [2]: https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#OpMemoryBarrier
