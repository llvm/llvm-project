; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

;; This test checks that the backend is capable to correctly translate
;; barrier OpenCL C 1.2 built-in function [1] into corresponding SPIR-V
;; instruction.

;; FIXME: Strictly speaking, this flag is not supported by barrier in OpenCL 1.2
;; #define CLK_IMAGE_MEM_FENCE 0x04
;;
;; void __attribute__((overloadable)) __attribute__((convergent)) barrier(cl_mem_fence_flags);
;;
;; __kernel void test_barrier_const_flags() {
;;   barrier(CLK_LOCAL_MEM_FENCE);
;;   barrier(CLK_GLOBAL_MEM_FENCE);
;;   barrier(CLK_IMAGE_MEM_FENCE);
;;
;;   barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
;;   barrier(CLK_LOCAL_MEM_FENCE | CLK_IMAGE_MEM_FENCE);
;;   barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE | CLK_IMAGE_MEM_FENCE);
;; }
;;
;; __kernel void test_barrier_non_const_flags(cl_mem_fence_flags flags) {
  ;; FIXME: OpenCL spec doesn't require flags to be compile-time known
  ;; barrier(flags);
;; }

; CHECK-SPIRV: OpName %[[#TEST_CONST_FLAGS:]] "test_barrier_const_flags"
; CHECK-SPIRV: %[[#UINT:]] = OpTypeInt 32 0

;; In SPIR-V, barrier is represented as OpControlBarrier [3] and OpenCL
;; cl_mem_fence_flags are represented as part of Memory Semantics [2], which
;; also includes memory order constraints. The backend applies some default
;; memory order for OpControlBarrier and therefore, constants below include a
;; bit more information than original source

;; 0x10 SequentiallyConsistent + 0x100 WorkgroupMemory
; CHECK-SPIRV:     %[[#LOCAL:]] = OpConstant %[[#UINT]] 272
;; 0x2 Workgroup
; CHECK-SPIRV:     %[[#WG:]] = OpConstant %[[#UINT]] 2
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

; CHECK-SPIRV: %[[#TEST_CONST_FLAGS]] = OpFunction %[[#]]
; CHECK-SPIRV: OpControlBarrier %[[#WG]] %[[#WG]] %[[#LOCAL]]
; CHECK-SPIRV: OpControlBarrier %[[#WG]] %[[#WG]] %[[#GLOBAL]]
; CHECK-SPIRV: OpControlBarrier %[[#WG]] %[[#WG]] %[[#IMAGE]]
; CHECK-SPIRV: OpControlBarrier %[[#WG]] %[[#WG]] %[[#LOCAL_GLOBAL]]
; CHECK-SPIRV: OpControlBarrier %[[#WG]] %[[#WG]] %[[#LOCAL_IMAGE]]
; CHECK-SPIRV: OpControlBarrier %[[#WG]] %[[#WG]] %[[#LOCAL_GLOBAL_IMAGE]]

define dso_local spir_kernel void @test_barrier_const_flags() local_unnamed_addr {
entry:
  tail call spir_func void @_Z7barrierj(i32 noundef 1)
  tail call spir_func void @_Z7barrierj(i32 noundef 2)
  tail call spir_func void @_Z7barrierj(i32 noundef 4)
  tail call spir_func void @_Z7barrierj(i32 noundef 3)
  tail call spir_func void @_Z7barrierj(i32 noundef 5)
  tail call spir_func void @_Z7barrierj(i32 noundef 7)
  ret void
}

declare spir_func void @_Z7barrierj(i32 noundef) local_unnamed_addr

define dso_local spir_kernel void @test_barrier_non_const_flags(i32 noundef %flags) local_unnamed_addr {
entry:
  ret void
}

;; References:
;; [1]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/barrier.html
;; [2]: https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#_a_id_memory_semantics__id_a_memory_semantics_lt_id_gt
;; [3]: https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#OpControlBarrier
