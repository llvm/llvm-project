; Test that the SPIR-V constructor/destructor lowering pass also runs for
; non-OpenMP compilations (e.g. direct `clang --target=spirv64-intel` builds of
; plain C++), and that llvm.global_ctors/llvm.global_dtors are not emitted as
; ordinary (and invalid) constant data.

; RUN: opt -S -passes=spirv-lower-ctor-dtor -mtriple=spirv64-intel-unknown < %s | FileCheck %s --check-prefix=IR
; RUN: llc -mtriple=spirv64-intel-unknown --spirv-ext=+SPV_INTEL_function_pointers %s -o - | FileCheck %s --check-prefix=SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-intel-unknown --spirv-ext=+SPV_INTEL_function_pointers %s -o - -filetype=obj | spirv-val %}

define void @my_ctor() addrspace(9) {
entry:
  ret void
}

define void @my_dtor() addrspace(9) {
entry:
  ret void
}

@llvm.global_ctors = appending addrspace(1) global [1 x { i32, ptr addrspace(9), ptr addrspace(9) }] [
  { i32, ptr addrspace(9), ptr addrspace(9) } { i32 65535, ptr addrspace(9) @my_ctor, ptr addrspace(9) null }
]

@llvm.global_dtors = appending addrspace(1) global [1 x { i32, ptr addrspace(9), ptr addrspace(9) }] [
  { i32, ptr addrspace(9), ptr addrspace(9) } { i32 65535, ptr addrspace(9) @my_dtor, ptr addrspace(9) null }
]

; IR-NOT: @llvm.global_ctors
; IR-NOT: @llvm.global_dtors
; IR: @__init_array_object_my_ctor_{{[a-f0-9]+}}_65535 = protected addrspace(1) constant ptr addrspace(9) @my_ctor, section ".init_array.65535"
; IR: @__init_array_start = weak protected addrspace(1) global ptr null
; IR: @__init_array_end = weak protected addrspace(1) global ptr null
; IR: @__fini_array_object_my_dtor_{{[a-f0-9]+}}_65535 = protected addrspace(1) constant ptr addrspace(9) @my_dtor, section ".fini_array.65535"
; IR: @__fini_array_start = weak protected addrspace(1) global ptr null
; IR: @__fini_array_end = weak protected addrspace(1) global ptr null
; IR: define weak_odr spir_kernel void @"spirv$device$init"()
; IR: define weak_odr spir_kernel void @"spirv$device$fini"()

; SPIRV-DAG: OpName %[[#]] "__init_array_object_my_ctor_{{[a-f0-9]+}}_65535"
; SPIRV-DAG: OpName %[[#]] "__fini_array_object_my_dtor_{{[a-f0-9]+}}_65535"
; SPIRV-DAG: OpEntryPoint Kernel %[[#]] "spirv$device$init"
; SPIRV-DAG: OpEntryPoint Kernel %[[#]] "spirv$device$fini"
; SPIRV-NOT: OpSpecConstantComposite

