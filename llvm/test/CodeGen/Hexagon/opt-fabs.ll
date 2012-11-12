; RUN: llc -mtriple=hexagon-unknown-elf -mcpu=hexagonv5  < %s | FileCheck %s
; Optimize fabsf to clrbit in V5.

; CHECK: r{{[0-9]+}} = clrbit(r{{[0-9]+}}, #31)

define float @my_fabsf(float %x) nounwind {
entry:
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  %0 = load float* %x.addr, align 4
  %call = call float @fabsf(float %0) readnone
  ret float %call
}

declare float @fabsf(float)
