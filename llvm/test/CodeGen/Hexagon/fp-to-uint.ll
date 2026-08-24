; RUN: llc --mtriple=hexagon -O2 -mattr=+hvxv79,+hvx-length128b %s -o - | FileCheck %s

define <16 x i1> @autogen_SD24352(<16 x float> %I8) {
BB:
  %FC10 = fptoui <16 x float> %I8 to <16 x i1>
  ret <16 x i1> %FC10
}

; CHECK:     r3:2 = combine(#1,##16843009)
; CHECK:     r1 = #124
; CHECK:     v1 = vxor(v1,v1)
; CHECK:     q0 = vcmp.eq(v0.sf,v1.sf)
; CHECK:     v0 = v1
; CHECK:     loop0([[LOOP:.LBB0_[0-9]+]],#32)
; CHECK:     v2 = vand(!q0,r2)
; CHECK:     r29 = and(r29,#-128)
; CHECK: [[LOOP]]:
; CHECK:     // =>This Inner Loop Header: Depth=1
; CHECK:     r1 = add(r1,#-4)
; CHECK:     v3 = vror(v2,r1)
; CHECK:     v0.w = vasl(v0.w,r3)
; CHECK:     v0 = vor(v0,v3)
; CHECK:     } :endloop0
; CHECK:     v2 = vsplat(r3)
; CHECK:     r1 = #-1
; CHECK:     memw(r29+#124) = r0
; CHECK:     v0 = vand(v0,v2)
; CHECK:     q0 = vcmp.eq(v0.w,v1.w)
; CHECK:     q0 = not(q0)
; CHECK:     v{{[0-9]+}} = vand(q0,r1)
; CHECK:     vmem(r29+#2) = v{{[0-9]+}}.new
