; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}

; TODO:  Implement support for the SPIR-V QuantizeToF16 operation
; XFAIL: *

; CHECK-DAG: [[F32:%.*]] = OpTypeFloat 32
; CHECK: %[[#]] = OpQuantizeToF16 [[F32]] %[[#]]
define spir_func void @test_wrappers() {
  entry:
  %r8 = call spir_func float @__spirv_QuantizeToF16(float 0.000000e+00)
  ret void
}

declare dso_local spir_func float @__spirv_QuantizeToF16(float)
