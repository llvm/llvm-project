; RUN: split-file %s %t
; RUN: not opt -S -dxil-resource-access -dxil-op-lower %t/texture2dms.ll 2>&1 | FileCheck %t/texture2dms.ll
; RUN: not opt -S -dxil-resource-access -dxil-op-lower %t/texture2dmsarray.ll 2>&1 | FileCheck %t/texture2dmsarray.ll
; RUN: not opt -S -dxil-resource-access -dxil-op-lower %t/texturecube.ll 2>&1 | FileCheck %t/texturecube.ll
; RUN: not opt -S -dxil-resource-access -dxil-op-lower %t/texturecubearray.ll 2>&1 | FileCheck %t/texturecubearray.ll

; The DXIL AtomicBinOp op only supports 1D, 2D, 3D and array textures, so
; multisampled and cube textures must be rejected.

;--- texture2dms.ll

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

;--- texture2dmsarray.ll

target triple = "dxil-pc-shadermodel6.6-compute"

; CHECK: DXIL atomicrmw not implemented for this texture resource kind
define i32 @atomic_texture2dmsarray(<3 x i32> %coords, i32 %value) {
  %texture = call target("dx.MSTexture", i32, 1, 4, 0, 8)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.MSTexture", i32, 1, 4, 0, 8) %texture, <3 x i32> %coords)
  %old = atomicrmw add ptr %ptr, i32 %value monotonic
  ret i32 %old
}

;--- texturecube.ll

target triple = "dxil-pc-shadermodel6.6-compute"

; CHECK: DXIL atomicrmw not implemented for this texture resource kind
define i32 @atomic_texturecube(<3 x i32> %coords, i32 %value) {
  %texture = call target("dx.Texture", i32, 1, 0, 0, 5)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.Texture", i32, 1, 0, 0, 5) %texture, <3 x i32> %coords)
  %old = atomicrmw umax ptr %ptr, i32 %value monotonic
  ret i32 %old
}

;--- texturecubearray.ll

target triple = "dxil-pc-shadermodel6.6-compute"

; CHECK: DXIL atomicrmw not implemented for this texture resource kind
define i32 @atomic_texturecubearray(<4 x i32> %coords, i32 %value) {
  %texture = call target("dx.Texture", i32, 1, 0, 0, 9)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.Texture", i32, 1, 0, 0, 9) %texture, <4 x i32> %coords)
  %old = atomicrmw xchg ptr %ptr, i32 %value monotonic
  ret i32 %old
}
