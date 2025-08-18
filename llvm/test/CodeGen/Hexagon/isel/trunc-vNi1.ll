; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; RUN: llc --mtriple=hexagon -mattr=+hvxv79,+hvx-length128b < %s | FileCheck %s

define void @f0(<2 x i32> %a0, ptr %a1) {
; CHECK-LABEL: f0:
; CHECK: r[[REG1H:([0-9]+)]]:[[REG1L:([0-9]+)]] = combine(#1,#1)
; CHECK: r[[REG2H:([0-9]+)]]:[[REG2L:([0-9]+)]] = and(r[[REG2H]]:[[REG2L]],r[[REG1H]]:[[REG1L]])
; CHECK: p{{[0-9]+}} = vcmpw.eq(r[[REG2H]]:[[REG2L]],#1)
b0:
  %v0 = trunc <2 x i32> %a0 to <2 x i1>
  store <2 x i1> %v0, ptr %a1, align 1
  ret void
}

define void @f1(<4 x i16> %a0, ptr %a1) {
; CHECK-LABEL: f1:
; CHECK: [[REG0:r([0-9]+)]] = and([[REG0]],##65537)
; CHECK: [[REG1:r([0-9]+)]] = and([[REG1]],##65537)
; CHECK: p{{[0-9]+}} = vcmph.eq(r{{[0-9]+}}:{{[0-9]+}},#1)
b0:
  %v0 = trunc <4 x i16> %a0 to <4 x i1>
  store <4 x i1> %v0, ptr %a1, align 1
  ret void
}

define void @f2(<8 x i8> %a0, ptr %a1) {
; CHECK-LABEL: f2:
; CHECK: [[REG0:r([0-9]+)]] = and([[REG0]],##16843009)
; CHECK: [[REG1:r([0-9]+)]] = and([[REG1]],##16843009)
; CHECK: p{{[0-9]+}} = vcmpb.eq(r{{[0-9]+}}:{{[0-9]+}},#1)
b0:
  %v0 = trunc <8 x i8> %a0 to <8 x i1>
  store <8 x i1> %v0, ptr %a1, align 1
  ret void
}

define void @f3(<4 x i8> %a0, ptr %a1) {
; CHECK-LABEL: f3:
; CHECK: r[[REGH:([0-9]+)]]:[[REGL:([0-9]+)]] = vzxtbh(r{{[0-9]+}})
; CHECK: r[[REGL]] = and(r[[REGL]],##65537)
; CHECK: r[[REGH]] = and(r[[REGH]],##65537)
; CHECK: p{{[0-9]+}} = vcmph.eq(r[[REGH]]:[[REGL]],#1)
b0:
  %v0 = trunc <4 x i8> %a0 to <4 x i1>
  store <4 x i1> %v0, ptr %a1, align 1
  ret void
}

define void @f4(<2 x i16> %a0, ptr %a1) {
; CHECK-LABEL: f4:
; CHECK: r[[REGH:([0-9]+)]]:[[REGL:([0-9]+)]] = vzxthw(r{{[0-9]+}})
; CHECK: r[[REG1H:([0-9]+)]]:[[REG1L:([0-9]+)]] = combine(#1,#1)
; CHECK: r[[REGH]]:[[REGL]] = and(r[[REGH]]:[[REGL]],r[[REG1H]]:[[REG1L]])
; CHECK: p{{[0-9]+}} = vcmpw.eq(r[[REGH]]:[[REGL]],#1)
b0:
  %v0 = trunc <2 x i16> %a0 to <2 x i1>
  store <2 x i1> %v0, ptr %a1, align 1
  ret void
}
