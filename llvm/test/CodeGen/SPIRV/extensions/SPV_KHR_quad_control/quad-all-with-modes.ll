; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv1.5-unknown-vulkan-pixel \
; RUN:   --spirv-ext=+SPV_KHR_quad_control,+SPV_KHR_maximal_reconvergence \
; RUN:   %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 \
; RUN:   -mtriple=spirv1.5-unknown-vulkan-pixel \
; RUN:   --spirv-ext=+SPV_KHR_quad_control,+SPV_KHR_maximal_reconvergence \
; RUN:   %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpCapability QuadControlKHR
; CHECK-DAG: OpExtension "SPV_KHR_quad_control"
; CHECK-DAG: OpExtension "SPV_KHR_maximal_reconvergence"
; CHECK: OpEntryPoint Fragment %[[#MAIN:]] "main"
; CHECK-DAG: OpExecutionMode %[[#MAIN]] QuadDerivativesKHR
; CHECK-DAG: OpExecutionMode %[[#MAIN]] RequireFullQuadsKHR
; CHECK-DAG: %[[#BOOL:]] = OpTypeBool
; CHECK-DAG: %[[#TRUE:]] = OpConstantTrue %[[#BOOL]]

; CHECK: %[[#MAIN]] = OpFunction
; CHECK: %[[#ALL:]] = OpGroupNonUniformQuadAllKHR %[[#BOOL]] %[[#TRUE]]
; CHECK: %[[#VALUE:]] = OpSelect %{{[0-9]+}} %[[#ALL]]
; CHECK: OpStore %{{[0-9]+}} %[[#VALUE]] Volatile
define void @main() #0 {
  %result = alloca i32
  %tok = call token @llvm.experimental.convergence.entry()
  %all = call i1 @llvm.spv.quad.all(i1 true) [ "convergencectrl"(token %tok) ]
  %value = zext i1 %all to i32
  store volatile i32 %value, ptr %result
  ret void
}

declare i1 @llvm.spv.quad.all(i1) convergent

attributes #0 = { convergent "hlsl.shader"="pixel" }

!spirv.ExecutionMode = !{!0, !1}
!0 = !{ptr @main, i32 5088}
!1 = !{ptr @main, i32 5089}
