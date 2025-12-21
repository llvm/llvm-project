; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv1.6-vulkan1.3-compute --spirv-ext=all %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-vulkan1.3-compute --spirv-ext=all %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK-NOT: OpExtension "SPV_KHR_no_integer_wrap_decoration"

define internal void @foo(i32 %i) local_unnamed_addr {
  %sub.i = sub nsw i32 0, %i
  ret void
}

define internal void @main() local_unnamed_addr #0 {
entry:
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }