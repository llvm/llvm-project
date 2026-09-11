; RUN: not opt -S -dxil-resource-access %s 2>&1 | FileCheck %s

; Verify that cmpxchg on a texture resource is rejected. DXIL has no texture
; atomic op, so the compare-exchange lowering must report an error.

target triple = "dxil-pc-shadermodel6.6-compute"

; CHECK: DXIL cmpxchg not implemented for texture resources

define void @cmpxchg_texture(i32 %index, i32 %cmp, i32 %value) {
  %texture = call target("dx.Texture", i32, 1, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.Texture", i32, 1, 0, 0, 2) %texture, i32 %index)
  %pair = cmpxchg ptr %ptr, i32 %cmp, i32 %value acq_rel monotonic
  ret void
}
