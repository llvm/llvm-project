; RUN: not opt -S -dxil-resource-access -dxil-op-lower %s 2>&1 | FileCheck %s

; The DXIL AtomicBinOp op only supports 1D, 2D, 3D and array textures.

target triple = "dxil-pc-shadermodel6.6-compute"

; CHECK: DXIL atomicrmw not implemented for this texture resource kind
define i32 @atomic_texture2dms(<2 x i32> %coords, i32 %value) {
  %texture = call target("dx.MSTexture", i32, 1, 4, 0, 3)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.MSTexture", i32, 1, 4, 0, 3) %texture, <2 x i32> %coords)
  %old = atomicrmw add ptr %ptr, i32 %value monotonic
  ret i32 %old
}
