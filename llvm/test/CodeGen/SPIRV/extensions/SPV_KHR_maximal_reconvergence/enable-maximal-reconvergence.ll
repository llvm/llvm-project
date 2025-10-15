; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv1.6-unknown-vulkan1.3-compute --spirv-ext=+SPV_KHR_maximal_reconvergence %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute --spirv-ext=+SPV_KHR_maximal_reconvergence %s -o - -filetype=obj | spirv-val %}

; CHECK: OpCapability Shader
; CHECK: OpExtension "SPV_KHR_maximal_reconvergence"
; CHECK: OpExecutionMode {{.*}} MaximallyReconvergesKHR
define void @main() local_unnamed_addr #0 {
entry:
  ret void
}

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) "enable-maximal-reconvergence"="true" "frame-pointer"="all" "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
