; RUN: llc --mtriple=hexagon -mattr=+hvxv79,+hvx-length128b < %s | FileCheck %s

; CHECK: [[REG0:r[0-9]+]] = memw(r{{[0-9]+}}+#0)
; CHECK: [[VREG1:v([0-9]+)]] = vsplat([[REG0]])
; CHECK: [[VREG2:v([0-9]+)]] = vand([[VREG1]],v{{[0-9]+}})
; CHECK: q[[QREG:[0-9]+]] =  vand([[VREG2]],r{{[0-9]+}})

define void @bitcast_v2i16_to_v32i1(ptr %in, ptr %out) {
entry:
  %load = load <2 x i16>, ptr %in, align 4
  %bitcast = bitcast <2 x i16> %load to <32 x i1>
  %extract = extractelement <32 x i1> %bitcast, i32 0
  %zext = zext i1 %extract to i8
  store i8 %zext, ptr %out, align 1
  ret void
}
