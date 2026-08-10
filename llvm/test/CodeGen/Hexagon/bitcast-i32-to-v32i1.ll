; RUN: llc --mtriple=hexagon -mattr=+hvxv79,+hvx-length128b < %s | FileCheck %s

; CHECK: [[VREG1:v([0-9]+)]] = vsplat(r{{[0-9]*}})
; CHECK: [[VREG2:v([0-9]+)]] = vand([[VREG1]],v{{[0-9]+}})
; CHECK: q[[QREG:[0-9]+]] =  vand([[VREG2]],r{{[0-9]+}})

define void @bitcast_i32_to_v32i1_full(ptr %in, ptr %out) {
entry:
  %load = load i32, ptr %in, align 4
  %bitcast = bitcast i32 %load to <32 x i1>
  %e0 = extractelement <32 x i1> %bitcast, i32 0
  %e1 = extractelement <32 x i1> %bitcast, i32 1
  %z0 = zext i1 %e0 to i8
  %z1 = zext i1 %e1 to i8
  %ptr0 = getelementptr i8, ptr %out, i32 0
  %ptr1 = getelementptr i8, ptr %out, i32 1
  store i8 %z0, ptr %ptr0, align 1
  store i8 %z1, ptr %ptr1, align 1
  ret void
}
