; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %s | FileCheck %s
; RUN: llc %s --filetype=obj -o - | obj2yaml | FileCheck %s --check-prefix=CHECK-OBJ

target triple = "dxil-pc-shadermodel6.7-library"

; Texture load and sample operations with non-constant (programmable) offsets
; require the AdvancedTextureOps shader feature flag.

; CHECK-OBJ: - Name: SFI0
; CHECK-OBJ:   Flags:
; CHECK-OBJ:     AdvancedTextureOps: true

; CHECK:      Combined Shader Flags for Module
; CHECK-NEXT: Shader Flags Value: 0x400000000

; CHECK: Note: shader requires additional functionality:
; CHECK:       Advanced Texture Ops

; CHECK: Function textureload_dynamic_offset : 0x400000000
define void @textureload_dynamic_offset(<2 x i32> %coords, <2 x i32> %offsets) #0 {
  %texture = call target("dx.Texture", <4 x float>, 0, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %data = call <4 x float> @llvm.dx.resource.load.level(
      target("dx.Texture", <4 x float>, 0, 0, 0, 2) %texture,
      <2 x i32> %coords, i32 0, <2 x i32> %offsets)
  ret void
}

; CHECK: Function textureload_const_offset : 0x00000000
define void @textureload_const_offset(<2 x i32> %coords) #0 {
  %texture = call target("dx.Texture", <4 x float>, 0, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %data = call <4 x float> @llvm.dx.resource.load.level(
      target("dx.Texture", <4 x float>, 0, 0, 0, 2) %texture,
      <2 x i32> %coords, i32 0, <2 x i32> <i32 1, i32 -1>)
  ret void
}

; CHECK: Function sample_dynamic_offset : 0x400000000
define void @sample_dynamic_offset(<2 x float> %coords, <2 x i32> %offsets) #0 {
  %texture = call target("dx.Texture", <4 x float>, 0, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %sampler = call target("dx.Sampler", 0)
      @llvm.dx.resource.handlefrombinding(i32 1, i32 0, i32 1, i32 0, ptr null)
  %data = call <4 x float> @llvm.dx.resource.sample(
      target("dx.Texture", <4 x float>, 0, 0, 0, 2) %texture,
      target("dx.Sampler", 0) %sampler,
      <2 x float> %coords, <2 x i32> %offsets)
  ret void
}

; CHECK: Function sample_clamp_dynamic_offset : 0x400000000
define void @sample_clamp_dynamic_offset(<2 x float> %coords, <2 x i32> %offsets,
                                         float %clamp) #0 {
  %texture = call target("dx.Texture", <4 x float>, 0, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %sampler = call target("dx.Sampler", 0)
      @llvm.dx.resource.handlefrombinding(i32 1, i32 0, i32 1, i32 0, ptr null)
  %data = call <4 x float> @llvm.dx.resource.sample.clamp(
      target("dx.Texture", <4 x float>, 0, 0, 0, 2) %texture,
      target("dx.Sampler", 0) %sampler,
      <2 x float> %coords, <2 x i32> %offsets, float %clamp)
  ret void
}

; CHECK: Function sample_const_offset : 0x00000000
define void @sample_const_offset(<2 x float> %coords) #0 {
  %texture = call target("dx.Texture", <4 x float>, 0, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %sampler = call target("dx.Sampler", 0)
      @llvm.dx.resource.handlefrombinding(i32 1, i32 0, i32 1, i32 0, ptr null)
  %data = call <4 x float> @llvm.dx.resource.sample(
      target("dx.Texture", <4 x float>, 0, 0, 0, 2) %texture,
      target("dx.Sampler", 0) %sampler,
      <2 x float> %coords, <2 x i32> zeroinitializer)
  ret void
}

; CHECK: Function samplebias_dynamic_offset : 0x400000000
define void @samplebias_dynamic_offset(<2 x float> %coords, float %bias,
                                       <2 x i32> %offsets) #0 {
  %texture = call target("dx.Texture", <4 x float>, 0, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %sampler = call target("dx.Sampler", 0)
      @llvm.dx.resource.handlefrombinding(i32 1, i32 0, i32 1, i32 0, ptr null)
  %data = call <4 x float> @llvm.dx.resource.samplebias(
      target("dx.Texture", <4 x float>, 0, 0, 0, 2) %texture,
      target("dx.Sampler", 0) %sampler,
      <2 x float> %coords, float %bias, <2 x i32> %offsets)
  ret void
}

; CHECK: Function samplebias_const_offset : 0x00000000
define void @samplebias_const_offset(<2 x float> %coords, float %bias) #0 {
  %texture = call target("dx.Texture", <4 x float>, 0, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %sampler = call target("dx.Sampler", 0)
      @llvm.dx.resource.handlefrombinding(i32 1, i32 0, i32 1, i32 0, ptr null)
  %data = call <4 x float> @llvm.dx.resource.samplebias(
      target("dx.Texture", <4 x float>, 0, 0, 0, 2) %texture,
      target("dx.Sampler", 0) %sampler,
      <2 x float> %coords, float %bias, <2 x i32> <i32 2, i32 3>)
  ret void
}

; CHECK: Function samplelevel_dynamic_offset : 0x400000000
define void @samplelevel_dynamic_offset(<2 x float> %coords, float %lod,
                                        <2 x i32> %offsets) #0 {
  %texture = call target("dx.Texture", <4 x float>, 0, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %sampler = call target("dx.Sampler", 0)
      @llvm.dx.resource.handlefrombinding(i32 1, i32 0, i32 1, i32 0, ptr null)
  %data = call <4 x float> @llvm.dx.resource.samplelevel(
      target("dx.Texture", <4 x float>, 0, 0, 0, 2) %texture,
      target("dx.Sampler", 0) %sampler,
      <2 x float> %coords, float %lod, <2 x i32> %offsets)
  ret void
}

; CHECK: Function samplelevel_const_offset : 0x00000000
define void @samplelevel_const_offset(<2 x float> %coords, float %lod) #0 {
  %texture = call target("dx.Texture", <4 x float>, 0, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %sampler = call target("dx.Sampler", 0)
      @llvm.dx.resource.handlefrombinding(i32 1, i32 0, i32 1, i32 0, ptr null)
  %data = call <4 x float> @llvm.dx.resource.samplelevel(
      target("dx.Texture", <4 x float>, 0, 0, 0, 2) %texture,
      target("dx.Sampler", 0) %sampler,
      <2 x float> %coords, float %lod, <2 x i32> zeroinitializer)
  ret void
}

; CHECK: Function samplegrad_dynamic_offset : 0x400000000
define void @samplegrad_dynamic_offset(<2 x float> %coords, <2 x float> %ddx,
                                       <2 x float> %ddy, <2 x i32> %offsets) #0 {
  %texture = call target("dx.Texture", <4 x float>, 0, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %sampler = call target("dx.Sampler", 0)
      @llvm.dx.resource.handlefrombinding(i32 1, i32 0, i32 1, i32 0, ptr null)
  %data = call <4 x float> @llvm.dx.resource.samplegrad(
      target("dx.Texture", <4 x float>, 0, 0, 0, 2) %texture,
      target("dx.Sampler", 0) %sampler,
      <2 x float> %coords, <2 x float> %ddx, <2 x float> %ddy,
      <2 x i32> %offsets)
  ret void
}

; CHECK: Function samplegrad_clamp_dynamic_offset : 0x400000000
define void @samplegrad_clamp_dynamic_offset(<2 x float> %coords,
                                             <2 x float> %ddx, <2 x float> %ddy,
                                             <2 x i32> %offsets,
                                             float %clamp) #0 {
  %texture = call target("dx.Texture", <4 x float>, 0, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %sampler = call target("dx.Sampler", 0)
      @llvm.dx.resource.handlefrombinding(i32 1, i32 0, i32 1, i32 0, ptr null)
  %data = call <4 x float> @llvm.dx.resource.samplegrad.clamp(
      target("dx.Texture", <4 x float>, 0, 0, 0, 2) %texture,
      target("dx.Sampler", 0) %sampler,
      <2 x float> %coords, <2 x float> %ddx, <2 x float> %ddy,
      <2 x i32> %offsets, float %clamp)
  ret void
}

; CHECK: Function samplegrad_const_offset : 0x00000000
define void @samplegrad_const_offset(<2 x float> %coords, <2 x float> %ddx,
                                     <2 x float> %ddy) #0 {
  %texture = call target("dx.Texture", <4 x float>, 0, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %sampler = call target("dx.Sampler", 0)
      @llvm.dx.resource.handlefrombinding(i32 1, i32 0, i32 1, i32 0, ptr null)
  %data = call <4 x float> @llvm.dx.resource.samplegrad(
      target("dx.Texture", <4 x float>, 0, 0, 0, 2) %texture,
      target("dx.Sampler", 0) %sampler,
      <2 x float> %coords, <2 x float> %ddx, <2 x float> %ddy,
      <2 x i32> <i32 -1, i32 1>)
  ret void
}

!llvm.module.flags = !{!0}
!dx.valver = !{!1}
!0 = !{i32 1, !"dx.resmayalias", i32 1}
!1 = !{i32 1, i32 8}

attributes #0 = { convergent norecurse nounwind "hlsl.export"}
