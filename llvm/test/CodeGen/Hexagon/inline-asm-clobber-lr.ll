; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: allocframe

target triple = "hexagon"

define internal fastcc void @f0() {
b0:
  %v0 = tail call i32* asm sideeffect "call 1f; r31.h = #hi(TH); r31.l = #lo(TH); jumpr r31; 1: $0 = r31", "=r,~{r28},~{r31}"()
  %v1 = bitcast i32* %v0 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 bitcast (void (...)* @f1 to i8*), i8* align 4 %v1, i32 12, i1 false)
  ret void
}

declare void @f1(...)

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture writeonly, i8* nocapture readonly, i32, i1) #0

attributes #0 = { argmemonly nounwind }
