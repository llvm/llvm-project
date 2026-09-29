; Check that protected visibility does not cause a crash, that protected
; declarations get Import linkage, and protected definitions get Export.

; RUN: split-file %s %t

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %t/opencl.ll -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %t/opencl.ll -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan1.3-compute %t/vulkan.ll -o - | FileCheck %s
; FIXME: re-enable validator check when spirv-val allows Linkage in vulkan env.
; RUNx: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute %t/vulkan.ll -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-vulkan-library %t/vulkan-lib.ll -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-vulkan-library %t/vulkan-lib.ll -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpName %[[#PROTECTED_DECL:]] "protected_decl"
; CHECK-DAG: OpName %[[#PROTECTED_DEF:]] "protected_def"

; CHECK-DAG: OpDecorate %[[#PROTECTED_DECL]] LinkageAttributes "protected_decl" Import
; CHECK-DAG: OpDecorate %[[#PROTECTED_DEF]] LinkageAttributes "protected_def" Export

;--- opencl.ll
declare protected spir_func void @protected_decl(ptr addrspace(1))

define protected spir_func void @protected_def(ptr addrspace(1) %x) {
entry:
  ret void
}

define spir_kernel void @test_kernel(ptr addrspace(1) %data) {
entry:
  call spir_func void @protected_decl(ptr addrspace(1) %data)
  call spir_func void @protected_def(ptr addrspace(1) %data)
  ret void
}

;--- vulkan.ll
declare protected void @protected_decl()

define protected void @protected_def() {
entry:
  ret void
}

define void @main() #0 {
entry:
  call void @protected_decl()
  call void @protected_def()
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }

;--- vulkan-lib.ll
declare protected void @protected_decl()

define protected void @protected_def() {
entry:
  ret void
}

define void @caller() {
entry:
  call void @protected_decl()
  call void @protected_def()
  ret void
}
