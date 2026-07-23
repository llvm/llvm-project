; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_bfloat16 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_bfloat16 %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpCapability BFloat16TypeKHR
; CHECK-DAG: OpCapability BFloat16DotProductKHR
; CHECK-DAG: OpExtension "SPV_KHR_bfloat16"
; CHECK: %[[#BFLOAT:]] = OpTypeFloat 16 0
; CHECK: %[[#]] = OpTypeVector %[[#BFLOAT]] 2
; CHECK: OpDot

declare spir_func bfloat @_Z3dotDv2_u6__bf16Dv2_S_(<2 x bfloat>, <2 x bfloat>)

define spir_kernel void @test() {
entry:
  %addrA = alloca <2 x bfloat>
  %addrB = alloca <2 x bfloat>
  %dataA = load <2 x bfloat>, ptr %addrA
  %dataB = load <2 x bfloat>, ptr %addrB
  %call = call spir_func bfloat @_Z3dotDv2_u6__bf16Dv2_S_(<2 x bfloat> %dataA, <2 x bfloat> %dataB)
  ret void
}
