; Test that the SPIR-V constructor/destructor lowering pass creates the
; expected init kernel and symbols for offload compilation.

; RUN: llc -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

define void @my_constructor() addrspace(4) {
entry:
  ret void
}

define void @my_destructor() addrspace(4) {
entry:
  ret void
}

define spir_kernel void @main() {
entry:
  ret void
}

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr}] [
  { i32, ptr, ptr } { i32 65535, ptr addrspacecast (ptr addrspace(4) @my_constructor to ptr), ptr null }
]

@llvm.global_dtors = appending global [1 x { i32, ptr, ptr}] [
  { i32, ptr, ptr } { i32 65535, ptr addrspacecast (ptr addrspace(4) @my_destructor to ptr), ptr null }
]

; This module has the openmp.is_target_device metadata to indicate offload mode.
!llvm.module.flags = !{!0}
!0 = !{i32 7, !"openmp-device", i32 50}

; CHECK-DAG: OpName %[[#INIT_KERNEL:]] "spirv$device$init"
; CHECK-DAG: OpName %[[#FINI_KERNEL:]] "spirv$device$fini"
; CHECK-DAG: OpName %[[#INIT_START:]] "__init_array_start"
; CHECK-DAG: OpName %[[#INIT_END:]] "__init_array_end"
; CHECK-DAG: OpName %[[#FINI_START:]] "__fini_array_start"
; CHECK-DAG: OpName %[[#FINI_END:]] "__fini_array_end"

; CHECK-DAG: OpName %[[#INIT_OBJ:]] "__init_array_object__{{[a-f0-9]+}}_65535"
; CHECK-DAG: OpName %[[#FINI_OBJ:]] "__fini_array_object__{{[a-f0-9]+}}_65535"

; CHECK: %[[#INIT_KERNEL]] = OpFunction
; CHECK: OpFunctionEnd

; CHECK: %[[#FINI_KERNEL]] = OpFunction
; CHECK: OpFunctionEnd
