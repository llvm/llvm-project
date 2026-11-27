; RUN: split-file %s %t
; RUN: cat %t/shader.ll %t/valver1.8.ll > %t/1.8.ll
; RUN: cat %t/shader.ll %t/valver1.7.ll > %t/1.7.ll
; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %t/1.8.ll | FileCheck %s --check-prefixes=CHECK,VALVER18
; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %t/1.7.ll | FileCheck %s --check-prefixes=CHECK,VALVER17
; RUN: llc %t/1.8.ll --filetype=obj -o - | obj2yaml | FileCheck %s --check-prefix=OBJ18
; RUN: llc %t/1.7.ll --filetype=obj -o - | obj2yaml | FileCheck %s --check-prefix=OBJ17

; Sample, SampleBias and SampleGrad with an instruction LOD clamp that is
; neither a constant zero nor undef require the TiledResources shader flag
; (0x1000). Matching DXC, the flag is only set for DXIL validator 1.8 or later.

; OBJ18: - Name: SFI0
; OBJ18:   Flags:
; OBJ18:     TiledResources: true

; No flags are set under validator 1.7, so SFI0 has no Flags entry.
; OBJ17:      - Name: SFI0
; OBJ17-NEXT:   Size: 8
; OBJ17-NEXT: - Name: HASH

; CHECK:         Combined Shader Flags for Module
; VALVER18-NEXT: Shader Flags Value: 0x00001000
; VALVER17-NEXT: Shader Flags Value: 0x00000000

; VALVER18:      Note: shader requires additional functionality:
; VALVER18:        Tiled resources
; VALVER17-NOT:  Tiled resources

;--- shader.ll
target triple = "dxil-pc-shadermodel6.7-library"

; VALVER18: Function sample_clamp_dynamic : 0x00001000
; VALVER17: Function sample_clamp_dynamic : 0x00000000
define void @sample_clamp_dynamic(<2 x float> %coords, float %clamp) #0 {
  %texture = call target("dx.Texture", <4 x float>, 0, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %sampler = call target("dx.Sampler", 0)
      @llvm.dx.resource.handlefrombinding(i32 1, i32 0, i32 1, i32 0, ptr null)
  %data = call <4 x float> @llvm.dx.resource.sample.clamp(
      target("dx.Texture", <4 x float>, 0, 0, 0, 2) %texture,
      target("dx.Sampler", 0) %sampler,
      <2 x float> %coords, <2 x i32> zeroinitializer, float %clamp)
  ret void
}

; VALVER18: Function sample_clamp_nonzero : 0x00001000
; VALVER17: Function sample_clamp_nonzero : 0x00000000
define void @sample_clamp_nonzero(<2 x float> %coords) #0 {
  %texture = call target("dx.Texture", <4 x float>, 0, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %sampler = call target("dx.Sampler", 0)
      @llvm.dx.resource.handlefrombinding(i32 1, i32 0, i32 1, i32 0, ptr null)
  %data = call <4 x float> @llvm.dx.resource.sample.clamp(
      target("dx.Texture", <4 x float>, 0, 0, 0, 2) %texture,
      target("dx.Sampler", 0) %sampler,
      <2 x float> %coords, <2 x i32> zeroinitializer, float 1.0)
  ret void
}

; CHECK: Function sample_clamp_zero : 0x00000000
define void @sample_clamp_zero(<2 x float> %coords) #0 {
  %texture = call target("dx.Texture", <4 x float>, 0, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %sampler = call target("dx.Sampler", 0)
      @llvm.dx.resource.handlefrombinding(i32 1, i32 0, i32 1, i32 0, ptr null)
  %data = call <4 x float> @llvm.dx.resource.sample.clamp(
      target("dx.Texture", <4 x float>, 0, 0, 0, 2) %texture,
      target("dx.Sampler", 0) %sampler,
      <2 x float> %coords, <2 x i32> zeroinitializer, float 0.0)
  ret void
}

; CHECK: Function sample_clamp_undef : 0x00000000
define void @sample_clamp_undef(<2 x float> %coords) #0 {
  %texture = call target("dx.Texture", <4 x float>, 0, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %sampler = call target("dx.Sampler", 0)
      @llvm.dx.resource.handlefrombinding(i32 1, i32 0, i32 1, i32 0, ptr null)
  %data = call <4 x float> @llvm.dx.resource.sample.clamp(
      target("dx.Texture", <4 x float>, 0, 0, 0, 2) %texture,
      target("dx.Sampler", 0) %sampler,
      <2 x float> %coords, <2 x i32> zeroinitializer, float undef)
  ret void
}

; VALVER18: Function samplebias_clamp_dynamic : 0x00001000
; VALVER17: Function samplebias_clamp_dynamic : 0x00000000
define void @samplebias_clamp_dynamic(<2 x float> %coords, float %bias,
                                      float %clamp) #0 {
  %texture = call target("dx.Texture", <4 x float>, 0, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %sampler = call target("dx.Sampler", 0)
      @llvm.dx.resource.handlefrombinding(i32 1, i32 0, i32 1, i32 0, ptr null)
  %data = call <4 x float> @llvm.dx.resource.samplebias.clamp(
      target("dx.Texture", <4 x float>, 0, 0, 0, 2) %texture,
      target("dx.Sampler", 0) %sampler,
      <2 x float> %coords, float %bias, <2 x i32> zeroinitializer,
      float %clamp)
  ret void
}

; CHECK: Function samplebias_clamp_zero : 0x00000000
define void @samplebias_clamp_zero(<2 x float> %coords, float %bias) #0 {
  %texture = call target("dx.Texture", <4 x float>, 0, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %sampler = call target("dx.Sampler", 0)
      @llvm.dx.resource.handlefrombinding(i32 1, i32 0, i32 1, i32 0, ptr null)
  %data = call <4 x float> @llvm.dx.resource.samplebias.clamp(
      target("dx.Texture", <4 x float>, 0, 0, 0, 2) %texture,
      target("dx.Sampler", 0) %sampler,
      <2 x float> %coords, float %bias, <2 x i32> zeroinitializer,
      float 0.0)
  ret void
}

; VALVER18: Function samplegrad_clamp_dynamic : 0x00001000
; VALVER17: Function samplegrad_clamp_dynamic : 0x00000000
define void @samplegrad_clamp_dynamic(<2 x float> %coords, <2 x float> %ddx,
                                      <2 x float> %ddy, float %clamp) #0 {
  %texture = call target("dx.Texture", <4 x float>, 0, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %sampler = call target("dx.Sampler", 0)
      @llvm.dx.resource.handlefrombinding(i32 1, i32 0, i32 1, i32 0, ptr null)
  %data = call <4 x float> @llvm.dx.resource.samplegrad.clamp(
      target("dx.Texture", <4 x float>, 0, 0, 0, 2) %texture,
      target("dx.Sampler", 0) %sampler,
      <2 x float> %coords, <2 x float> %ddx, <2 x float> %ddy,
      <2 x i32> zeroinitializer, float %clamp)
  ret void
}

; CHECK: Function samplegrad_clamp_undef : 0x00000000
define void @samplegrad_clamp_undef(<2 x float> %coords, <2 x float> %ddx,
                                    <2 x float> %ddy) #0 {
  %texture = call target("dx.Texture", <4 x float>, 0, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %sampler = call target("dx.Sampler", 0)
      @llvm.dx.resource.handlefrombinding(i32 1, i32 0, i32 1, i32 0, ptr null)
  %data = call <4 x float> @llvm.dx.resource.samplegrad.clamp(
      target("dx.Texture", <4 x float>, 0, 0, 0, 2) %texture,
      target("dx.Sampler", 0) %sampler,
      <2 x float> %coords, <2 x float> %ddx, <2 x float> %ddy,
      <2 x i32> zeroinitializer, float undef)
  ret void
}

attributes #0 = { convergent norecurse nounwind "hlsl.export"}

;--- valver1.8.ll
!dx.valver = !{!0}
!0 = !{i32 1, i32 8}

;--- valver1.7.ll
!dx.valver = !{!0}
!0 = !{i32 1, i32 7}
