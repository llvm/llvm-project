; DEFINE: %{quad-exts} = +SPV_KHR_quad_control,+SPV_KHR_maximal_reconvergence
; DEFINE: %{deriv-ext} = +SPV_KHR_compute_shader_derivatives

; RUN: llc -O0 -verify-machineinstrs \
; RUN:   -mtriple=spirv1.5-unknown-vulkan-compute \
; RUN:   --spirv-ext=%{deriv-ext},%{quad-exts} \
; RUN:   %s -o - | FileCheck %s --check-prefixes=CHECK,NATIVE
; RUN: llc -O0 -verify-machineinstrs \
; RUN:   -mtriple=spirv1.5-unknown-vulkan-compute \
; RUN:   --spirv-ext=%{deriv-ext},-SPV_KHR_quad_control \
; RUN:   %s -o - | FileCheck %s --check-prefixes=CHECK,EMULATED \
; RUN:     --implicit-check-not=SPV_KHR_quad_control \
; RUN:     --implicit-check-not=SPV_KHR_maximal_reconvergence
; RUN: %if spirv-tools %{ llc -O0 \
; RUN:   -mtriple=spirv1.5-unknown-vulkan-compute \
; RUN:   --spirv-ext=%{deriv-ext},%{quad-exts} \
; RUN:   %s -o - -filetype=obj | spirv-val --target-env vulkan1.2 %}
; RUN: %if spirv-tools %{ llc -O0 \
; RUN:   -mtriple=spirv1.5-unknown-vulkan-compute \
; RUN:   --spirv-ext=%{deriv-ext},-SPV_KHR_quad_control \
; RUN:   %s -o - -filetype=obj | spirv-val --target-env vulkan1.2 %}

; CHECK-DAG: OpCapability ComputeDerivativeGroupQuadsKHR
; NATIVE-DAG: OpCapability QuadControlKHR
; CHECK-DAG: OpExtension "SPV_KHR_compute_shader_derivatives"
; NATIVE-DAG: OpExtension "SPV_KHR_quad_control"
; NATIVE-DAG: OpExtension "SPV_KHR_maximal_reconvergence"
; CHECK: OpEntryPoint GLCompute %[[#MAIN:]] "main"
; CHECK-DAG: OpExecutionMode %[[#MAIN]] DerivativeGroupQuadsKHR
; CHECK-DAG: OpExecutionMode %[[#MAIN]] LocalSize 2 2 1
; CHECK-DAG: %[[#BOOL:]] = OpTypeBool
; CHECK-DAG: %[[#TRUE:]] = OpConstantTrue %[[#BOOL]]

; CHECK: %[[#MAIN]] = OpFunction
; NATIVE: %[[#ALL:]] = OpGroupNonUniformQuadAllKHR %[[#BOOL]] %[[#TRUE]]
; EMULATED-COUNT-3: OpGroupNonUniformQuadSwap %[[#BOOL]] %{{[0-9]+}} %[[#TRUE]]
; EMULATED-NEXT: %[[#ALL:]] = OpLogicalAnd %[[#BOOL]]
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

attributes #0 = {
  convergent "hlsl.numthreads"="2,2,1" "hlsl.shader"="compute"
}

!spirv.ExecutionMode = !{!0}
!0 = !{ptr @main, i32 5289}
