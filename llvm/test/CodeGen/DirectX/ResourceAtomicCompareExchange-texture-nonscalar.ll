; RUN: split-file %s %t
; RUN: not opt -S -dxil-resource-access -dxil-op-lower %t/texture1d-int2.ll 2>&1 | FileCheck %t/texture1d-int2.ll
; RUN: not opt -S -dxil-resource-access -dxil-op-lower %t/texture2d-int4.ll 2>&1 | FileCheck %t/texture2d-int4.ll
; RUN: not opt -S -dxil-resource-access -dxil-op-lower %t/texture2darray-i64x2.ll 2>&1 | FileCheck %t/texture2darray-i64x2.ll
; RUN: not opt -S -dxil-resource-access -dxil-op-lower %t/texture3d-float4.ll 2>&1 | FileCheck %t/texture3d-float4.ll

; A texture atomic operates on a whole texel, so there is no way to address a
; single component of a multi-component texel.

;--- texture1d-int2.ll

target triple = "dxil-pc-shadermodel6.6-compute"

; CHECK: DXIL cmpxchg requires a texture resource with a scalar element type
define i32 @cmpxchg_texture1d_int2(i32 %coord, i32 %cmp, i32 %value) {
  %texture = call target("dx.Texture", <2 x i32>, 1, 0, 0, 1)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.Texture", <2 x i32>, 1, 0, 0, 1) %texture, i32 %coord)
  %pair = cmpxchg ptr %ptr, i32 %cmp, i32 %value monotonic monotonic
  %old = extractvalue { i32, i1 } %pair, 0
  ret i32 %old
}

;--- texture2d-int4.ll

target triple = "dxil-pc-shadermodel6.6-compute"

; CHECK: DXIL cmpxchg requires a texture resource with a scalar element type
define i32 @cmpxchg_texture2d_int4(<2 x i32> %coords, i32 %cmp, i32 %value) {
  %texture = call target("dx.Texture", <4 x i32>, 1, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.Texture", <4 x i32>, 1, 0, 0, 2) %texture, <2 x i32> %coords)
  %pair = cmpxchg ptr %ptr, i32 %cmp, i32 %value monotonic monotonic
  %old = extractvalue { i32, i1 } %pair, 0
  ret i32 %old
}

;--- texture2darray-i64x2.ll

target triple = "dxil-pc-shadermodel6.6-compute"

; CHECK: DXIL cmpxchg requires a texture resource with a scalar element type
define i64 @cmpxchg_texture2darray_i64x2(<3 x i32> %coords, i64 %cmp,
                                         i64 %value) {
  %texture = call target("dx.Texture", <2 x i64>, 1, 0, 0, 7)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.Texture", <2 x i64>, 1, 0, 0, 7) %texture, <3 x i32> %coords)
  %pair = cmpxchg ptr %ptr, i64 %cmp, i64 %value monotonic monotonic
  %old = extractvalue { i64, i1 } %pair, 0
  ret i64 %old
}

;--- texture3d-float4.ll

target triple = "dxil-pc-shadermodel6.6-compute"

; CHECK: DXIL cmpxchg requires a texture resource with a scalar element type
define i32 @cmpxchg_texture3d_float4(<3 x i32> %coords, i32 %cmp, i32 %value) {
  %texture = call target("dx.Texture", <4 x float>, 1, 0, 0, 4)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.Texture", <4 x float>, 1, 0, 0, 4) %texture, <3 x i32> %coords)
  %pair = cmpxchg ptr %ptr, i32 %cmp, i32 %value monotonic monotonic
  %old = extractvalue { i32, i1 } %pair, 0
  ret i32 %old
}
