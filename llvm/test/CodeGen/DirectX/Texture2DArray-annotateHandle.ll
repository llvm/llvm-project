; RUN: opt -S -passes=dxil-translate-metadata,dxil-op-lower %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.6-pixel"

declare void @use_float4(<4 x float>)

; Test that a Texture2DArray handle gets ResourceKind::Texture2DArray metadata.
; CHECK-LABEL: define void @annotate_texture2darray_float4(
define void @annotate_texture2darray_float4(<3 x float> %coords, float %bias) {
  %texture = call target("dx.Texture", <4 x float>, 0, 0, 0, 7)
      @llvm.dx.resource.handlefrombinding.tdx.Texture_v4f32_0_0_0_7t(
          i32 0, i32 5, i32 1, i32 0, ptr null)
  %sampler = call target("dx.Sampler", 0)
      @llvm.dx.resource.handlefrombinding.tdx.Sampler_0t(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK: %[[TEX:.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217, %dx.types.ResBind { i32 5, i32 5, i32 0, i8 0 }, i32 5, i1 false)
  ; CHECK: %[[TEX_ANNOT:.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle %[[TEX]], %dx.types.ResourceProperties { i32 7, i32 1033 })
  ; CHECK: call %dx.types.ResRet.f32 @dx.op.sampleBias.f32(i32 61, %dx.types.Handle %[[TEX_ANNOT]]
  %data = call <4 x float>
      @llvm.dx.resource.samplebias.v4f32.tdx.Texture_v4f32_0_0_0_7t.tdx.Sampler_0t.v3f32.v2i32(
          target("dx.Texture", <4 x float>, 0, 0, 0, 7) %texture,
          target("dx.Sampler", 0) %sampler,
          <3 x float> %coords, float %bias, <2 x i32> zeroinitializer)

  call void @use_float4(<4 x float> %data)
  ret void
}

declare target("dx.Texture", <4 x float>, 0, 0, 0, 7)
    @llvm.dx.resource.handlefrombinding.tdx.Texture_v4f32_0_0_0_7t(i32, i32, i32, i32, ptr)
declare target("dx.Sampler", 0)
    @llvm.dx.resource.handlefrombinding.tdx.Sampler_0t(i32, i32, i32, i32, ptr)
declare <4 x float>
    @llvm.dx.resource.samplebias.v4f32.tdx.Texture_v4f32_0_0_0_7t.tdx.Sampler_0t.v3f32.v2i32(
        target("dx.Texture", <4 x float>, 0, 0, 0, 7),
        target("dx.Sampler", 0), <3 x float>, float, <2 x i32>)
