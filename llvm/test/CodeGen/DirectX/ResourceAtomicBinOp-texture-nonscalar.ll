; RUN: not opt -S -dxil-resource-access -dxil-op-lower %s 2>&1 | FileCheck %s

; A texture atomic operates on a whole texel, so there is no way to address a
; single component of a multi-component texel.

target triple = "dxil-pc-shadermodel6.6-compute"

; CHECK: DXIL atomicrmw requires a texture resource with a scalar integer element type
define i32 @atomic_texture2d_int4(<2 x i32> %coords, i32 %value) {
  %texture = call target("dx.Texture", <4 x i32>, 1, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.Texture", <4 x i32>, 1, 0, 0, 2) %texture, <2 x i32> %coords)
  %old = atomicrmw add ptr %ptr, i32 %value monotonic
  ret i32 %old
}
