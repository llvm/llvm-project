; RUN: llc -O0 -mtriple=spirv1.6-vulkan1.3-compute %s -o - | FileCheck %s --check-prefix=CHECK-NOEXT
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-vulkan1.3-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; RUN: llc -O0 -mtriple=spirv1.6-vulkan1.3-compute -spirv-ext=+SPV_KHR_float_controls2 %s -o - | FileCheck %s --check-prefix=CHECK-EXT
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-vulkan1.3-compute -spirv-ext=+SPV_KHR_float_controls2 %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK-NOEXT-NOT: OpDecorate FPFastMathMode

; CHECK-EXT: OpCapability FloatControls2
; CHECK-EXT: OpExtension "SPV_KHR_float_controls2"
; CHECK-EXT: OpDecorate {{%[0-9]+}} FPFastMathMode NotNaN|NotInf|NSZ|AllowRecip|Fast

define hidden spir_func float @foo(float  %0) local_unnamed_addr {
  %2 = fmul reassoc nnan ninf nsz arcp afn float %0, 2.000000e+00
  ret float %2
}

define void @main() local_unnamed_addr #1 {
  ret void
}

attributes #1 = { "hlsl.numthreads"="8,1,1" "hlsl.shader"="compute" }
