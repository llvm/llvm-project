; RUN: llc -mtriple=hexagon-unknown-elf -mcpu=hexagonv5 -hexagon-bit=0 < %s | FileCheck %s
; Optimize fabsf to clrbit in V5.

; CHECK: r{{[0-9]+}} = clrbit(r{{[0-9]+}},#31)

define float @my_llvm.fabs.f32(float %x) nounwind {
entry:
  %x.addr = alloca float, align 4
  store float %x, ptr %x.addr, align 4
  %0 = load float, ptr %x.addr, align 4
  %call = call float @llvm.fabs.f32(float %0) readnone
  ret float %call
}

declare float @llvm.fabs.f32(float)
