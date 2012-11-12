; RUN: llc -march=hexagon -mcpu=hexagonv5 < %s | FileCheck %s
; Optimize fneg to togglebit in V5.

define float @foo(float %x) nounwind {
entry:
; CHECK: r{{[0-9]+}} = togglebit(r{{[0-9]+}}, #31)
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  %0 = load float* %x.addr, align 4
  %sub = fsub float -0.000000e+00, %0
  ret float %sub
}

define float @bar(float %x) nounwind {
entry:
; CHECK: r{{[0-9]+}} = togglebit(r{{[0-9]+}}, #31)
  %sub = fsub float -0.000000e+00, %x
  ret float %sub
}

define float @baz(float %x) nounwind {
entry:
; CHECK: r{{[0-9]+}} = togglebit(r{{[0-9]+}}, #31)
  %conv1 = fmul float %x, -1.000000e+00
  ret float %conv1
}
