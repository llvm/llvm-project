; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_bfloat16 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_bfloat16 %s -o - -filetype=obj | spirv-val %}
; XFAIL: *
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-ERROR

; CHECK-ERROR: BFloat16TypeKHR requires the following SPIR-V extension: SPV_KHR_subgroup_rotate

; CHECK-DAG: OpCapability BFloat16TypeKHR
; CHECK-DAG: OpExtension "SPV_KHR_bfloat16"
; CHECK: %[[#BFLOAT:]] = OpTypeFloat 16 0
; CHECK: %[[#]] = OpTypeVector %[[#BFLOAT]] 2

define spir_kernel void @test() {
entry:
  %addr1 = alloca bfloat
  ret void
}
