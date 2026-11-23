; Test to ensure the SPIR-V constructor/destructor lowering pass is not run
; when OpenMP offload metadata is absent.

; RUN: llc -mtriple=spirv64-intel-unknown --spirv-ext=+SPV_INTEL_function_pointers %s -o /dev/null -print-after=spirv-lower-ctor-dtor 2>&1 | FileCheck %s

define void @my_ctor() addrspace(9) {
entry:
  ret void
}

define void @my_dtor() addrspace(9) {
entry:
  ret void
}

@llvm.global_ctors = appending global [1 x { i32, ptr addrspace(9), ptr addrspace(9) }] [
  { i32, ptr addrspace(9), ptr addrspace(9) } { i32 100, ptr addrspace(9) @my_ctor, ptr addrspace(9) null }
]

@llvm.global_dtors = appending global [1 x { i32, ptr addrspace(9), ptr addrspace(9) }] [
  { i32, ptr addrspace(9), ptr addrspace(9) } { i32 100, ptr addrspace(9) @my_dtor, ptr addrspace(9) null }
]

; Verify that NO init/fini kernels or symbols are created
; CHECK-NOT: @__init_array_object_high_priority_{{[a-f0-9]+}}_100
; CHECK-NOT: @__init_array_object_low_priority_{{[a-f0-9]+}}_50
; CHECK-NOT: @__init_array_start = weak protected addrspace(1)
; CHECK-NOT: @__init_array_end = weak protected addrspace(1)

; CHECK-NOT: @__fini_array_object_high_priority_{{[a-f0-9]+}}_100
; CHECK-NOT: @__fini_array_object_low_priority_{{[a-f0-9]+}}_50
; CHECK-NOT: @__fini_array_start = weak protected addrspace(1)
; CHECK-NOT: @__fini_array_end = weak protected addrspace(1)

; CHECK-NOT: define weak_odr spir_kernel void @"spirv$device$init"()
; CHECK-NOT: define weak_odr spir_kernel void @"spirv$device$fini"()
