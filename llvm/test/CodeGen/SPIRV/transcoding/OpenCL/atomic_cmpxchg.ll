; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

;; This test checks that the backend is capable to correctly translate
;; atomic_cmpxchg OpenCL C 1.2 built-in function [1] into corresponding SPIR-V
;; instruction.

;; __kernel void test_atomic_cmpxchg(__global int *p, int cmp, int val) {
;;   atomic_cmpxchg(p, cmp, val);
;;
;;   __global unsigned int *up = (__global unsigned int *)p;
;;   unsigned int ucmp = (unsigned int)cmp;
;;   unsigned int uval = (unsigned int)val;
;;   atomic_cmpxchg(up, ucmp, uval);
;; }

; CHECK-SPIRV:     OpName %[[#TEST:]] "test_atomic_cmpxchg"
; CHECK-SPIRV-DAG: %[[#UINT:]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG: %[[#UINT_PTR:]] = OpTypePointer CrossWorkgroup %[[#UINT]]

;; In SPIR-V, atomic_cmpxchg is represented as OpAtomicCompareExchange [2],
;; which also includes memory scope and two memory semantic arguments. The
;; backend applies some default memory order for it and therefore, constants
;; below include a bit more information than original source

;; 0x2 Workgroup
; CHECK-SPIRV-DAG: %[[#WORKGROUP_SCOPE:]] = OpConstant %[[#UINT]] 2

;; 0x0 Relaxed
;; TODO: do we need CrossWorkgroupMemory here as well?
; CHECK-SPIRV-DAG: %[[#RELAXED:]] = OpConstant %[[#UINT]] 0

; CHECK-SPIRV:     %[[#TEST]] = OpFunction %[[#]]
; CHECK-SPIRV:     %[[#PTR:]] = OpFunctionParameter %[[#UINT_PTR]]
; CHECK-SPIRV:     %[[#CMP:]] = OpFunctionParameter %[[#UINT]]
; CHECK-SPIRV:     %[[#VAL:]] = OpFunctionParameter %[[#UINT]]
; CHECK-SPIRV:     %[[#]] = OpAtomicCompareExchange %[[#UINT]] %[[#PTR]] %[[#WORKGROUP_SCOPE]] %[[#RELAXED]] %[[#RELAXED]] %[[#VAL]] %[[#CMP]]
; CHECK-SPIRV:     %[[#]] = OpAtomicCompareExchange %[[#UINT]] %[[#PTR]] %[[#WORKGROUP_SCOPE]] %[[#RELAXED]] %[[#RELAXED]] %[[#VAL]] %[[#CMP]]

define dso_local spir_kernel void @test_atomic_cmpxchg(i32 addrspace(1)* noundef %p, i32 noundef %cmp, i32 noundef %val) local_unnamed_addr {
entry:
  %call = tail call spir_func i32 @_Z14atomic_cmpxchgPU3AS1Viii(i32 addrspace(1)* noundef %p, i32 noundef %cmp, i32 noundef %val)
  %call1 = tail call spir_func i32 @_Z14atomic_cmpxchgPU3AS1Vjjj(i32 addrspace(1)* noundef %p, i32 noundef %cmp, i32 noundef %val)
  ret void
}

declare spir_func i32 @_Z14atomic_cmpxchgPU3AS1Viii(i32 addrspace(1)* noundef, i32 noundef, i32 noundef) local_unnamed_addr

declare spir_func i32 @_Z14atomic_cmpxchgPU3AS1Vjjj(i32 addrspace(1)* noundef, i32 noundef, i32 noundef) local_unnamed_addr

;; References:
;; [1]: https://www.khronos.org/registry/OpenCL/sdk/2.0/docs/man/xhtml/atomic_cmpxchg.html
;; [2]: https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#OpAtomicCompareExchange
