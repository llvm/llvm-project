; Verify that FPFastMathMode decorations are emitted on OpFunctionCall
; when SPV_KHR_float_controls2 is enabled.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_float_controls2 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_float_controls2 %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-NO-DECO
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpDecorate %[[#CALL_RES:]] FPFastMathMode NotNaN|NotInf|NSZ|AllowRecip|AllowContract|AllowReassoc|AllowTransform
; CHECK: %[[#CALL_RES]] = OpFunctionCall %[[#]] %[[#]]

; CHECK-NO-DECO-NOT: FPFastMathMode

define internal spir_func float @helper(float %x) {
  %r = fmul float %x, %x
  ret float %r
}

define spir_kernel void @test(ptr addrspace(1) %out, float %a) {
entry:
  %call_fast = call fast spir_func float @helper(float %a)
  store float %call_fast, ptr addrspace(1) %out, align 4
  ret void
}
