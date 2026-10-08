; Test that the SPIR-V constructor/destructor lowering pass creates the
; expected init kernel and symbols for offload compilation.

; RUN: llc -mtriple=spirv64-intel-unknown --spirv-ext=+SPV_INTEL_function_pointers %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-intel-unknown --spirv-ext=+SPV_INTEL_function_pointers %s -o - -filetype=obj | spirv-val %}

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

; This module has the openmp.is_target_device metadata to indicate offload mode.
!llvm.module.flags = !{!0}
!0 = !{i32 7, !"openmp-device", i32 50}

; CHECK-DAG: OpName %[[#INIT_KERNEL:]] "spirv$device$init"
; CHECK-DAG: OpName %[[#FINI_KERNEL:]] "spirv$device$fini"
; CHECK-DAG: OpName %[[#INIT_START:]] "__init_array_start"
; CHECK-DAG: OpName %[[#INIT_END:]] "__init_array_end"
; CHECK-DAG: OpName %[[#FINI_START:]] "__fini_array_start"
; CHECK-DAG: OpName %[[#FINI_END:]] "__fini_array_end"

; CHECK-DAG: OpName %[[#INIT_OBJ:]] "__init_array_object_my_constructor_{{[a-f0-9]+}}_65535"
; CHECK-DAG: OpName %[[#FINI_OBJ:]] "__fini_array_object_my_destructor_{{[a-f0-9]+}}_65535"

; CHECK: %[[#INIT_KERNEL]] = OpFunction
; CHECK: %{{[0-9]+}} = OpBitcast %{{[0-9]+}} %[[#INIT_START]]
; CHECK: %{{[0-9]+}} = OpBitcast %{{[0-9]+}} %[[#INIT_END]]
; CHECK: OpFunctionEnd

; CHECK: %[[#FINI_KERNEL]] = OpFunction
; CHECK: %{{[0-9]+}} = OpBitcast %{{[0-9]+}} %[[#FINI_START]]
; CHECK: %{{[0-9]+}} = OpBitcast %{{[0-9]+}} %[[#FINI_END]]
; CHECK: OpFunctionEnd
