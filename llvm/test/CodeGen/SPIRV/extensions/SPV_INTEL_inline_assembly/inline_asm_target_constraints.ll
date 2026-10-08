; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-amd-amdhsa %s --spirv-ext=+SPV_INTEL_inline_assembly -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s --spirv-ext=+SPV_INTEL_inline_assembly -o - -filetype=obj | spirv-val %}

; Test that SPIR-V backend handles target-specific inline asm constraints
; (e.g., AMDGPU's "v" for VGPR) without crashing.

; CHECK-DAG: OpCapability AsmINTEL
; CHECK-DAG: OpExtension "SPV_INTEL_inline_assembly"
; CHECK-DAG: %[[#Int32Ty:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Int64Ty:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#Fun:]] = OpTypeFunction %[[#Int64Ty]] %[[#Int32Ty]]
; CHECK-DAG: %[[#Dialect:]] = OpAsmTargetINTEL "spirv64-amd-amdhsa"
; CHECK-DAG: %[[#Asm:]] = OpAsmINTEL %[[#Int64Ty]] %[[#Fun]] %[[#Dialect]] "v_mov_b32 $0, $1" "=v,v"

; CHECK: OpFunction
; CHECK: OpAsmCallINTEL %[[#Int64Ty]] %[[#Asm]] %[[#]]

define i64 @test_vgpr_constraint(i32 %x) {
  %res = call i64 asm sideeffect "v_mov_b32 $0, $1", "=v,v"(i32 %x)
  ret i64 %res
}
