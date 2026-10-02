; Verify that FPFastMathMode decorations are emitted for OpFunctionCall when 
; SPV_KHR_float_controls2 is enabled. Also verify that non-core instructions
; (e.g. OpGroupFMulKHR from SPV_KHR_uniform_group_instructions)do NOT get a 
; spurious FPFastMathMode decoration. SPV_KHR_float_controls2 extends
; FPFastMathMode to all *core* instructions, but not to extension-defined ones.

; Without FC2: no decoration on call.
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_uniform_group_instructions %s -o - | FileCheck %s --check-prefix=NO-DECO
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_uniform_group_instructions %s -o - -filetype=obj | spirv-val %}

; With FC2: decoration emitted on OpFunctionCall, not on extension instructions.
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_float_controls2,+SPV_KHR_uniform_group_instructions %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_float_controls2,+SPV_KHR_uniform_group_instructions %s -o - -filetype=obj | spirv-val %}

; NO-DECO-NOT: FPFastMathMode

; CHECK-DAG: OpDecorate %[[#CALL_RES:]] FPFastMathMode NotNaN|NotInf|NSZ|AllowRecip|AllowContract|AllowReassoc|AllowTransform
; CHECK-DAG: %[[#CALL_RES]] = OpFunctionCall %[[#]] %[[#]]
; CHECK-DAG: %[[#GFMUL_RES:]] = OpGroupFMulKHR

; OpGroupFMulKHR is defined by SPV_KHR_uniform_group_instructions, not core.
; It should NOT get FPFastMathMode because FC2 only covers core instructions.
; CHECK-NOT: OpDecorate %[[#GFMUL_RES]] FPFastMathMode

define internal spir_func float @helper(float %x) {
  %r = fmul float %x, %x
  ret float %r
}

declare spir_func float @_Z20__spirv_GroupFMulKHR(i32, i32, float)

define spir_kernel void @test(ptr addrspace(1) %out, float %a) {
entry:
  %call_fast = call fast spir_func float @helper(float %a)
  store float %call_fast, ptr addrspace(1) %out, align 4

  %gfmul_fast = call fast spir_func float @_Z20__spirv_GroupFMulKHR(i32 2, i32 0, float %a)
  store float %gfmul_fast, ptr addrspace(1) %out, align 4

  ret void
}
