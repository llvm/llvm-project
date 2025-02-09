; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; CHECK: allocframe

target triple = "hexagon"

define internal fastcc void @f0() {
b0:
  %v0 = tail call ptr asm sideeffect "call 1f; r31.h = #hi(TH); r31.l = #lo(TH); jumpr r31; 1: $0 = r31", "=r,~{r28},~{r31}"()
  call void @llvm.memcpy.p0.p0.i32(ptr align 4 @f1, ptr align 4 %v0, i32 12, i1 false)
  ret void
}

declare void @f1(...)

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0.p0.i32(ptr nocapture writeonly, ptr nocapture readonly, i32, i1) #0

attributes #0 = { argmemonly nounwind }
