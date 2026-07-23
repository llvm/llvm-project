; Confirm that we handle fpmath metadata correctly
; This is a copy of https://github.com/KhronosGroup/SPIRV-LLVM-Translator/blob/main/test/extensions/INTEL/SPV_INTEL_fp_max_error/IntelFPMaxErrorFPMath.ll

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_INTEL_fp_max_error %s -o %t.spt
; RUN: FileCheck %s --input-file=%t.spt
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_INTEL_fp_max_error %s -o - -filetype=obj | spirv-val %}

; CHECK: OpCapability FPMaxErrorINTEL
; CHECK: OpExtension "SPV_INTEL_fp_max_error"

; CHECK: OpName %[[#CalleeName:]] "callee"
; CHECK: OpDecorate %[[#F3:]] FPMaxErrorDecorationINTEL 1075838976
; CHECK: OpDecorate %[[#Callee:]] FPMaxErrorDecorationINTEL 1065353216

; CHECK: %[[#FloatTy:]] = OpTypeFloat 32

define float @callee(float %f1, float %f2) {
entry:
ret float %f1
}

; CHECK: %[[#F3]] = OpFDiv %[[#FloatTy]]
; CHECK: %[[#Callee]] = OpFunctionCall %[[#FloatTy]] %[[#CalleeName]]

define void @test_fp_max_error_decoration(float %f1, float %f2, ptr %out) {
entry:
%f3 = fdiv float %f1, %f2, !fpmath !0
store volatile float %f3, ptr %out
%call = call float @callee(float %f1, float %f2), !fpmath !1
store volatile float %call, ptr %out
ret void
}

!0 = !{float 2.500000e+00}
!1 = !{float 1.000000e+00}
