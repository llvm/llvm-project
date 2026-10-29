; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv1.5-unknown-vulkan-library \
; RUN:   --spirv-ext=+SPV_KHR_quad_control,+SPV_KHR_maximal_reconvergence \
; RUN:   %s -o - | FileCheck %s
; RUN: not llc -O0 -mtriple=spirv1.5-unknown-vulkan-library \
; RUN:   %s -o - 2>&1 | FileCheck %s --check-prefix=NO-EXT
; RUN: not llc -O0 -mtriple=spirv1.5-unknown-vulkan-library \
; RUN:   --spirv-ext=+SPV_KHR_quad_control,-SPV_KHR_maximal_reconvergence \
; RUN:   %s -o - 2>&1 | FileCheck %s --check-prefix=NO-RECONVERGENCE
; RUN: not llc -O0 -mtriple=spirv32v1.2-unknown-vulkan-library \
; RUN:   --spirv-ext=+SPV_KHR_quad_control,+SPV_KHR_maximal_reconvergence \
; RUN:   %s -o - 2>&1 | FileCheck %s --check-prefix=VERSION
; RUN: %if spirv-tools %{ llc -O0 \
; RUN:   -mtriple=spirv1.5-unknown-vulkan-library \
; RUN:   --spirv-ext=+SPV_KHR_quad_control,+SPV_KHR_maximal_reconvergence \
; RUN:   %s -o - -filetype=obj | spirv-val %}

; NO-EXT: OpGroupNonUniformQuadAllKHR instruction requires the following SPIR-V extension: SPV_KHR_quad_control
; NO-RECONVERGENCE: SPV_KHR_quad_control requires the following SPIR-V extension: SPV_KHR_maximal_reconvergence
; VERSION: OpGroupNonUniformQuadAllKHR requires SPIR-V 1.3

; CHECK-DAG: OpCapability QuadControlKHR
; CHECK-DAG: OpExtension "SPV_KHR_quad_control"
; CHECK-DAG: OpExtension "SPV_KHR_maximal_reconvergence"
; CHECK-DAG: %[[#BOOL:]] = OpTypeBool

; CHECK-LABEL: Begin function quad_all
; CHECK: %[[#COND:]] = OpFunctionParameter %[[#BOOL]]
; CHECK: %[[#RET:]] = OpGroupNonUniformQuadAllKHR %[[#BOOL]] %[[#COND]]
; CHECK-NEXT: OpReturnValue %[[#RET]]
define i1 @quad_all(i1 %cond) convergent {
  %tok = call token @llvm.experimental.convergence.entry()
  %ret = call i1 @llvm.spv.quad.all(i1 %cond) [ "convergencectrl"(token %tok) ]
  ret i1 %ret
}

declare i1 @llvm.spv.quad.all(i1) convergent
