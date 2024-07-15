; Adapted from https://github.com/KhronosGroup/SPIRV-LLVM-Translator/tree/main/test/extensions/INTEL/SPV_INTEL_global_variable_fpga_decorations

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_global_variable_fpga_decorations %s -o - | FileCheck %s --check-prefixes=CHECK-SPIRV
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_global_variable_fpga_decorations %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV: Capability GlobalVariableFPGADecorationsINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_global_variable_fpga_decorations"
; CHECK-SPIRV-DAG: OpName %[[#G1:]] "int_var"
; CHECK-SPIRV-DAG: OpName %[[#G2:]] "float_var"
; CHECK-SPIRV-DAG: OpName %[[#G3:]] "bool_var"
; CHECK-SPIRV-DAG: OpDecorate %[[#G1]] ImplementInRegisterMapINTEL 1
; CHECK-SPIRV-DAG: OpDecorate %[[#G1]] InitModeINTEL 0
; CHECK-SPIRV-DAG: OpDecorate %[[#G2]] ImplementInRegisterMapINTEL 1
; CHECK-SPIRV-DAG: OpDecorate %[[#G2]] InitModeINTEL 1
; CHECK-SPIRV-DAG: OpDecorate %[[#G3]] ImplementInRegisterMapINTEL 0
; CHECK-SPIRV-DAG: OpDecorate %[[#G3]] InitModeINTEL 0

@int_var = addrspace(1) global i32 42, !spirv.Decorations !1
@float_var = addrspace(1) global float 1.0, !spirv.Decorations !5
@bool_var = addrspace(1) global i1 0, !spirv.Decorations !7

define spir_kernel void @test() {
entry:
  ret void
}

!1 = !{!2, !3}
!2 = !{i32 6191, i1 true} ; ImplementInRegisterMapINTEL = true
!3 = !{i32 6190, i32 0} ; InitModeINTEL = 0
!4 = !{i32 6190, i32 1} ; InitModeINTEL = 1
!5 = !{!2, !4}
!6 = !{i32 6191, i1 false} ; ImplementInRegisterMapINTEL = false
!7 = !{!6, !3}
