; Adapted from https://github.com/KhronosGroup/SPIRV-LLVM-Translator/tree/main/test/extensions/INTEL/SPV_INTEL_global_variable_host_access

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_global_variable_host_access,+SPV_INTEL_global_variable_fpga_decorations %s -o - | FileCheck %s --check-prefixes=CHECK-SPIRV
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_global_variable_host_access,+SPV_INTEL_global_variable_fpga_decorations %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV-DAG: Capability GlobalVariableHostAccessINTEL
; CHECK-SPIRV-DAG: Capability GlobalVariableFPGADecorationsINTEL
; CHECK-SPIRV-DAG: Extension "SPV_INTEL_global_variable_host_access"
; CHECK-SPIRV-DAG: Extension "SPV_INTEL_global_variable_fpga_decorations"

; CHECK-SPIRV-DAG: OpName %[[#G1:]] "int_var"
; CHECK-SPIRV-DAG: OpName %[[#G2:]] "bool_var"
; CHECK-SPIRV-DAG: OpName %[[#G3:]] "float_var"
; CHECK-SPIRV-DAG: OpDecorate %[[#G1]] HostAccessINTEL 1 "IntVarName"
; CHECK-SPIRV-DAG: OpDecorate %[[#G2]] HostAccessINTEL 3 "BoolVarName"
; CHECK-SPIRV-DAG: OpDecorate %[[#G3]] ImplementInRegisterMapINTEL 1
; CHECK-SPIRV-DAG: OpDecorate %[[#G3]] InitModeINTEL 1

@int_var = addrspace(1) global i32 42, !spirv.Decorations !1
@bool_var = addrspace(1) global i1 0, !spirv.Decorations !4
@float_var = addrspace(1) global float 1.0, !spirv.Decorations !5

define spir_kernel void @test() {
entry:
  ret void
}

!1 = !{!2}
!2 = !{i32 6188, i32 1, !"IntVarName"} ; HostAccessINTEL 1 "IntVarName"
!3 = !{i32 6188, i32 3, !"BoolVarName"} ; HostAccessINTEL 3 "BoolVarName"
!4 = !{!3}
!5 = !{!6, !7}
!6 = !{i32 6191, i1 true} ; ImplementInRegisterMapINTEL = true
!7 = !{i32 6190, i32 1} ; InitModeINTEL = 1
