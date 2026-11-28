; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv1.5-unknown-vulkan-library \
; RUN:   --spirv-ext=+SPV_KHR_quad_control,+SPV_KHR_maximal_reconvergence \
; RUN:   %s -o - | FileCheck %s
; RUN: llc -O0 -verify-machineinstrs \
; RUN:   -mtriple=spirv32v1.3-unknown-vulkan-library %s -o - \
; RUN:   | FileCheck %s --check-prefix=EMULATED \
; RUN:     --implicit-check-not=OpGroupNonUniformQuadAllKHR \
; RUN:     --implicit-check-not=OpExtension
; RUN: llc -O0 -verify-machineinstrs \
; RUN:   -mtriple=spirv1.5-unknown-vulkan-library \
; RUN:   --spirv-ext=-SPV_KHR_quad_control,-SPV_KHR_maximal_reconvergence \
; RUN:   %s -o - | FileCheck %s --check-prefix=EMULATED \
; RUN:     --implicit-check-not=OpGroupNonUniformQuadAllKHR \
; RUN:     --implicit-check-not=OpExtension
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
; RUN: %if spirv-tools %{ llc -O0 \
; RUN:   -mtriple=spirv32v1.3-unknown-vulkan-library \
; RUN:   %s -o - -filetype=obj | spirv-val %}

; NO-RECONVERGENCE: SPV_KHR_quad_control requires the following SPIR-V extension: SPV_KHR_maximal_reconvergence
; VERSION: OpGroupNonUniformQuadAllKHR requires SPIR-V 1.3

; CHECK-DAG: OpCapability QuadControlKHR
; CHECK-DAG: OpExtension "SPV_KHR_quad_control"
; CHECK-DAG: OpExtension "SPV_KHR_maximal_reconvergence"
; CHECK-NOT: OpExecutionMode
; CHECK-DAG: %[[#BOOL:]] = OpTypeBool

; CHECK-LABEL: Begin function quad_all
; CHECK: %[[#COND:]] = OpFunctionParameter %[[#BOOL]]
; CHECK: %[[#RET:]] = OpGroupNonUniformQuadAllKHR %[[#BOOL]] %[[#COND]]
; CHECK-NEXT: OpReturnValue %[[#RET]]

; EMULATED-DAG: OpCapability GroupNonUniformQuad
; EMULATED-NOT: OpExecutionMode
; EMULATED-DAG: %[[#BOOL:]] = OpTypeBool
; EMULATED-DAG: %[[#INT:]] = OpTypeInt 32 0
; EMULATED-DAG: %[[#SCOPE:]] = OpConstant %[[#INT]] 3
; EMULATED-DAG: %[[#XDIR:]] = OpConstant %[[#INT]] 0
; EMULATED-DAG: %[[#YDIR:]] = OpConstant %[[#INT]] 1
; EMULATED-DAG: %[[#DDIR:]] = OpConstant %[[#INT]] 2
; EMULATED-LABEL: Begin function quad_all
; EMULATED: %[[#COND:]] = OpFunctionParameter %[[#BOOL]]
; EMULATED: %[[#X:]] = OpGroupNonUniformQuadSwap %[[#BOOL]] %[[#SCOPE]] %[[#COND]] %[[#XDIR]]
; EMULATED-NEXT: %[[#ANDX:]] = OpLogicalAnd %[[#BOOL]] %[[#COND]] %[[#X]]
; EMULATED-NEXT: %[[#Y:]] = OpGroupNonUniformQuadSwap %[[#BOOL]] %[[#SCOPE]] %[[#COND]] %[[#YDIR]]
; EMULATED-NEXT: %[[#ANDY:]] = OpLogicalAnd %[[#BOOL]] %[[#ANDX]] %[[#Y]]
; EMULATED-NEXT: %[[#D:]] = OpGroupNonUniformQuadSwap %[[#BOOL]] %[[#SCOPE]] %[[#COND]] %[[#DDIR]]
; EMULATED-NEXT: %[[#RET:]] = OpLogicalAnd %[[#BOOL]] %[[#ANDY]] %[[#D]]
; EMULATED-NEXT: OpReturnValue %[[#RET]]
define i1 @quad_all(i1 %cond) convergent {
  %tok = call token @llvm.experimental.convergence.entry()
  %ret = call i1 @llvm.spv.quad.all(i1 %cond) [ "convergencectrl"(token %tok) ]
  ret i1 %ret
}

declare i1 @llvm.spv.quad.all(i1) convergent
