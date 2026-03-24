; Test to ensure the SPIR-V constructor/destructor lowering pass is not run
; when OpenMP offload metadata is absent.

; RUN: llc -mtriple=spirv64-intel-unknown --spirv-ext=+SPV_INTEL_function_pointers %s -o - | FileCheck %s

define void @my_constructor() addrspace(9) {
entry:
  ret void
}

define void @my_destructor() addrspace(9) {
entry:
  ret void
}

@llvm.global_ctors = appending global [1 x { i32, ptr addrspace(9), ptr addrspace(9) }] [
  { i32, ptr addrspace(9), ptr addrspace(9) } { i32 65535, ptr addrspace(9) @my_constructor, ptr addrspace(9) null }
]

@llvm.global_dtors = appending global [1 x { i32, ptr addrspace(9), ptr addrspace(9) }] [
  { i32, ptr addrspace(9), ptr addrspace(9) } { i32 65535, ptr addrspace(9) @my_destructor, ptr addrspace(9) null }
]

; CHECK-NOT: OpName %{{[0-9]+}} "spirv$device$init"
; CHECK-NOT: OpName %{{[0-9]+}} "spirv$device$fini"
; CHECK-NOT: OpName %{{[0-9]+}} "__init_array_start"
; CHECK-NOT: OpName %{{[0-9]+}} "__init_array_end"
; CHECK-NOT: OpName %{{[0-9]+}} "__fini_array_start"
; CHECK-NOT: OpName %{{[0-9]+}} "__fini_array_end"
; CHECK-NOT: __init_array_object_
; CHECK-NOT: __fini_array_object_

; Verify the original constructors/destructors still exist
; CHECK-DAG: OpName %[[#CTOR:]] "my_constructor"
; CHECK-DAG: OpName %[[#DTOR:]] "my_destructor"

; CHECK: %[[#CTOR]] = OpFunction
; CHECK: OpFunctionEnd

; CHECK: %[[#DTOR]] = OpFunction
; CHECK: OpFunctionEnd
