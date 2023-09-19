;; Sources:
;;
;; void kernel foo(__read_only image2d_t src) {
;;   sampler_t sampler1 = CLK_NORMALIZED_COORDS_TRUE |
;;                        CLK_ADDRESS_REPEAT |
;;                        CLK_FILTER_NEAREST;
;;   sampler_t sampler2 = 0x00;
;;
;;   read_imagef(src, sampler1, 0, 0);
;;   read_imagef(src, sampler2, 0, 0);
;; }

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV: %[[#SamplerID0:]] = OpConstantSampler %[[#]] Repeat 1 Nearest
; CHECK-SPIRV: %[[#SamplerID1:]] = OpConstantSampler %[[#]] None 0 Nearest
; CHECK-SPIRV: %[[#]] = OpSampledImage %[[#]] %[[#]] %[[#SamplerID0]]
; CHECK-SPIRV: %[[#]] = OpSampledImage %[[#]] %[[#]] %[[#SamplerID1]]

define spir_func <4 x float> @foo(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %src) local_unnamed_addr {
entry:
  %0 = tail call target("spirv.Sampler") @__translate_sampler_initializer(i32 23)
  %1 = tail call target("spirv.Sampler") @__translate_sampler_initializer(i32 0)
  %call = tail call spir_func <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv2_ff(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %src, target("spirv.Sampler") %0, <2 x float> zeroinitializer, float 0.000000e+00)
  %call1 = tail call spir_func <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv2_ff(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %src, target("spirv.Sampler") %1, <2 x float> zeroinitializer, float 0.000000e+00)
  %add = fadd <4 x float> %call, %call1
  ret <4 x float> %add
}

declare target("spirv.Sampler") @__translate_sampler_initializer(i32) local_unnamed_addr

declare spir_func <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv2_ff(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0), target("spirv.Sampler"), <2 x float>, float) local_unnamed_addr
