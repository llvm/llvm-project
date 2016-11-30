; RUN: opt -amdgpu-clp-vector-expansion -mtriple=amdgcn--amdhsa -mcpu=fiji -S < %s | FileCheck %s
; Function Attrs: nounwind
define spir_func <2 x float> @foo(<2 x float> %p) #0 {
entry:
; CHECK: %lo.call = call spir_func float @_Z4cbrtf(float
; CHECK: insertelement <2 x float> undef, float %lo.call, i32 0
; CHECK: %hi.call = call spir_func float @_Z4cbrtf(float
  %call = call spir_func <2 x float> @_Z4cbrtDv2_f(<2 x float> %p) #1
  ret <2 x float> %call
}

; Function Attrs: nounwind readnone
declare spir_func <2 x float> @_Z4cbrtDv2_f(<2 x float>) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
