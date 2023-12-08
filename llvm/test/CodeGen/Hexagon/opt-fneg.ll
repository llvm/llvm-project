; RUN: llc -march=hexagon -mcpu=hexagonv5 < %s | FileCheck %s
; Optimize fneg to togglebit in V5.

define float @foo(float %x) nounwind {
entry:
; CHECK-LABEL: foo:
; CHECK: r{{[0-9]+}} = togglebit(r{{[0-9]+}},#31)
  %x.addr = alloca float, align 4
  store float %x, ptr %x.addr, align 4
  %0 = load float, ptr %x.addr, align 4
  %sub = fsub float -0.000000e+00, %0
  ret float %sub
}

define float @bar(float %x) nounwind {
entry:
; CHECK-LABEL: bar:
; CHECK: r{{[0-9]+}} = togglebit(r{{[0-9]+}},#31)
  %sub = fsub float -0.000000e+00, %x
  ret float %sub
}

define float @baz0(float %x) nounwind {
entry:
; CHECK-LABEL: baz0:
; CHECK: r{{[0-9]+}} = togglebit(r{{[0-9]+}},#31)
  %conv1 = fmul nnan float %x, -1.000000e+00
  ret float %conv1
}

define float @baz1(float %x) nounwind {
entry:
  %not.nan = fadd nnan float %x, %x
; CHECK-LABEL: baz1:
; CHECK: r{{[0-9]+}} = togglebit(r{{[0-9]+}},#31)
  %conv1 = fmul float %not.nan, -1.000000e+00
  ret float %conv1
}
