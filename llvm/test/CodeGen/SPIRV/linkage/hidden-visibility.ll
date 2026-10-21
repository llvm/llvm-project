; Check that hidden visibility does not cause a crash and that hidden
; declarations get Import linkage while hidden definitions do not.

; RUN: split-file %s %t

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %t/opencl.ll -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %t/opencl.ll -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan1.3-compute %t/vulkan.ll -o - | FileCheck %s
; FIXME: re-enable validator check when spirv-val allows Linkage in vulkan env.
; RUNx: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute %t/vulkan.ll -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-vulkan-library %t/vulkan-lib.ll -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-vulkan-library %t/vulkan-lib.ll -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpName %[[#HIDDEN_HELPER:]] "hidden_helper"
; CHECK-DAG: OpName %[[#HIDDEN_DEF:]] "hidden_def"

; CHECK-DAG: OpDecorate %[[#HIDDEN_HELPER]] LinkageAttributes "hidden_helper" Import
; CHECK-NOT: OpDecorate %[[#HIDDEN_DEF]] LinkageAttributes

;--- opencl.ll
@hidden_leaf_var = external hidden addrspace(1) global i32

declare hidden spir_func void @hidden_helper(ptr addrspace(1))

define hidden spir_func void @hidden_def(ptr addrspace(1) %x) {
entry:
  ret void
}

define spir_kernel void @test_kernel(ptr addrspace(1) %data) {
entry:
  %val = load i32, ptr addrspace(1) @hidden_leaf_var
  store i32 %val, ptr addrspace(1) %data
  call spir_func void @hidden_helper(ptr addrspace(1) %data)
  call spir_func void @hidden_def(ptr addrspace(1) %data)
  ret void
}

;--- vulkan.ll
declare hidden void @hidden_helper()

define hidden void @hidden_def() {
entry:
  ret void
}

define void @main() #0 {
entry:
  call void @hidden_helper()
  call void @hidden_def()
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }

;--- vulkan-lib.ll
declare hidden void @hidden_helper()

define hidden void @hidden_def() {
entry:
  ret void
}

define void @caller() {
entry:
  call void @hidden_helper()
  call void @hidden_def()
  ret void
}
