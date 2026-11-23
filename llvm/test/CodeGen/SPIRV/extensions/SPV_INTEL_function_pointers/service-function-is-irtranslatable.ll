; Ensure that implicitly inserted service functions are lowered into valid MIR.
; RUN: llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers %s -o - | FileCheck %s
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers %s -o - | FileCheck %s
; TODO: We do not generate correct SPIR-V for this pattern, which is used for e.g. std::endl handling, thus spirv-val is disabled.
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers %s -o - -filetype=obj | spirv-val %}
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers %s -o - -filetype=obj | spirv-val %}

; CHECK: OpCapability FunctionPointersINTEL
; CHECK: OpExtension "SPV_INTEL_function_pointers"
; CHECK: OpDecorate %[[#]] LinkageAttributes "e" Import

declare spir_func align 8 ptr addrspace(4) @e()

define spir_func ptr @f() {
entry:
  %0 = load ptr, ptr @e, align 8
  ret ptr %0
}
