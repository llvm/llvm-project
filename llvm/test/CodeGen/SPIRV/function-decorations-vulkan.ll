; RUN: llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; Use a decoration valid for Shader modules to check that function metadata
; is also lowered for Vulkan. RelaxedPrecision applies to the return value.
; Repeated metadata entries must produce only one decoration.

; CHECK: OpName %[[#Helper:]] "helper"
; CHECK-NOT: RelaxedPrecision
; CHECK: OpDecorate %[[#Helper]] RelaxedPrecision
; CHECK-NOT: RelaxedPrecision
; CHECK: %[[#Helper]] = OpFunction

define internal float @helper(float %x) noinline !spirv.Decorations !0 {
  ret float %x
}

define void @main() #0 {
  %result = call float @helper(float 1.0)
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }

!0 = !{!1, !1}
!1 = !{i32 0} ; RelaxedPrecision
