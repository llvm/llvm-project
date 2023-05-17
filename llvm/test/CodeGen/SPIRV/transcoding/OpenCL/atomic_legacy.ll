; RUN: llc -O0 -opaque-pointers=0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

;; This test checks that the backend is capable to correctly translate
;; legacy atomic OpenCL C 1.2 built-in functions [1] into corresponding SPIR-V
;; instruction.

;; __kernel void test_legacy_atomics(__global int *p, int val) {
;;   atom_add(p, val);     // from cl_khr_global_int32_base_atomics
;;   atomic_add(p, val);   // from OpenCL C 1.1
;; }

; CHECK-SPIRV:     OpName %[[#TEST:]] "test_legacy_atomics"
; CHECK-SPIRV-DAG: %[[#UINT:]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG: %[[#UINT_PTR:]] = OpTypePointer CrossWorkgroup %[[#UINT]]

;; In SPIR-V, atomic_add is represented as OpAtomicIAdd [2], which also includes
;; memory scope and memory semantic arguments. The backend applies a default
;; memory scope and memory order for it and therefore, constants below include
;; a bit more information than original source

;; 0x2 Workgroup
; CHECK-SPIRV-DAG: %[[#WORKGROUP_SCOPE:]] = OpConstant %[[#UINT]] 2

;; 0x0 Relaxed
; CHECK-SPIRV-DAG: %[[#RELAXED:]] = OpConstant %[[#UINT]] 0

; CHECK-SPIRV:     %[[#TEST]] = OpFunction %[[#]]
; CHECK-SPIRV:     %[[#PTR:]] = OpFunctionParameter %[[#UINT_PTR]]
; CHECK-SPIRV:     %[[#VAL:]] = OpFunctionParameter %[[#UINT]]
; CHECK-SPIRV:     %[[#]] = OpAtomicIAdd %[[#UINT]] %[[#PTR]] %[[#WORKGROUP_SCOPE]] %[[#RELAXED]] %[[#VAL]]
; CHECK-SPIRV:     %[[#]] = OpAtomicIAdd %[[#UINT]] %[[#PTR]] %[[#WORKGROUP_SCOPE]] %[[#RELAXED]] %[[#VAL]]

define dso_local spir_kernel void @test_legacy_atomics(i32 addrspace(1)* noundef %p, i32 noundef %val) local_unnamed_addr {
entry:
  %call = tail call spir_func i32 @_Z8atom_addPU3AS1Vii(i32 addrspace(1)* noundef %p, i32 noundef %val)
  %call1 = tail call spir_func i32 @_Z10atomic_addPU3AS1Vii(i32 addrspace(1)* noundef %p, i32 noundef %val)
  ret void
}

declare spir_func i32 @_Z8atom_addPU3AS1Vii(i32 addrspace(1)* noundef, i32 noundef) local_unnamed_addr

declare spir_func i32 @_Z10atomic_addPU3AS1Vii(i32 addrspace(1)* noundef, i32 noundef) local_unnamed_addr

;; References:
;; [1]: https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_C.html#atomic-legacy
;; [2]: https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#OpAtomicIAdd
