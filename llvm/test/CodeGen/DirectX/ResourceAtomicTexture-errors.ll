; RUN: not opt -S -dxil-resource-access %s 2>&1 | FileCheck %s

; Verify that atomic operations on texture resources are rejected. DXIL has no
; texture atomic op, so both atomicrmw and cmpxchg must report an error.

target triple = "dxil-pc-shadermodel6.6-compute"

; CHECK: DXIL atomicrmw not implemented for texture resources

define void @atomicrmw_texture(i32 %index, i32 %value) {
  %texture = call target("dx.Texture", i32, 1, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.Texture", i32, 1, 0, 0, 2) %texture, i32 %index)
  %old = atomicrmw add ptr %ptr, i32 %value acq_rel
  ret void
}
