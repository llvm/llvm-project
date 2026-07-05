; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv1.6-unknown-vulkan1.3-compute --spirv-ext=+SPV_KHR_maximal_reconvergence %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute --spirv-ext=+SPV_KHR_maximal_reconvergence %s -o - -filetype=obj | spirv-val %}

; CHECK: OpCapability Shader
; CHECK: OpExtension "SPV_KHR_maximal_reconvergence"
; CHECK-NOT: OpExecutionMode {{.*}} MaximallyReconvergesKHR
; CHECK: OpExecutionMode [[main:%[0-9]+]] MaximallyReconvergesKHR
; CHECK-NOT: OpExecutionMode {{.*}} MaximallyReconvergesKHR
; CHECK: OpName [[main]] "main"
define void @main() local_unnamed_addr #0 {
entry:
  ret void
}

define void @negative() local_unnamed_addr #1 {
entry:
  ret void
}

attributes #0 = { "enable-maximal-reconvergence"="true" "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
attributes #1 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
