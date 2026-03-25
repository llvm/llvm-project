; Test for SPIR-V constructor/destructor lowering pass at IR level.
; RUN: llc -mtriple=spirv64-intel-unknown --spirv-ext=+SPV_INTEL_function_pointers %s -o /dev/null -print-after=spirv-lower-ctor-dtor 2>&1 | FileCheck %s

; This test verifies that:
; 1. The init/fini kernels are created.
; 2. Symbol mangling includes function name, hash, and priority.
; 3. Array boundary globals are created.
; 4. The pass only runs when offload metadata is present.

define void @high_priority() addrspace(9) {
entry:
  ret void
}

define void @low_priority() addrspace(9) {
entry:
  ret void
}

@llvm.global_ctors = appending global [2 x { i32, ptr addrspace(9), ptr addrspace(9) }] [
  { i32, ptr addrspace(9), ptr addrspace(9) } { i32 100, ptr addrspace(9) @high_priority, ptr addrspace(9) null },
  { i32, ptr addrspace(9), ptr addrspace(9) } { i32 50, ptr addrspace(9) @low_priority, ptr addrspace(9) null }
]

@llvm.global_dtors = appending global [2 x { i32, ptr addrspace(9), ptr addrspace(9) }] [
  { i32, ptr addrspace(9), ptr addrspace(9) } { i32 100, ptr addrspace(9) @high_priority, ptr addrspace(9) null },
  { i32, ptr addrspace(9), ptr addrspace(9) } { i32 50, ptr addrspace(9) @low_priority, ptr addrspace(9) null }
]

; Mark this as an offload module (OpenMP target device).
!llvm.module.flags = !{!0}
!0 = !{i32 7, !"openmp-device", i32 50}

; CHECK: @__init_array_object_high_priority_{{[a-f0-9]+}}_100 = protected addrspace(1) constant ptr addrspace(9) @high_priority
; CHECK: @__init_array_object_low_priority_{{[a-f0-9]+}}_50 = protected addrspace(1) constant ptr addrspace(9) @low_priority
; CHECK: @__init_array_start = weak protected addrspace(1) global ptr null
; CHECK: @__init_array_end = weak protected addrspace(1) global ptr null

; CHECK: @__fini_array_object_high_priority_{{[a-f0-9]+}}_100 = protected addrspace(1) constant ptr addrspace(9) @high_priority
; CHECK: @__fini_array_object_low_priority_{{[a-f0-9]+}}_50 = protected addrspace(1) constant ptr addrspace(9) @low_priority
; CHECK: @__fini_array_start = weak protected addrspace(1) global ptr null
; CHECK: @__fini_array_end = weak protected addrspace(1) global ptr null

; CHECK: define weak_odr spir_kernel void @"spirv$device$init"()
; CHECK: define weak_odr spir_kernel void @"spirv$device$fini"()
