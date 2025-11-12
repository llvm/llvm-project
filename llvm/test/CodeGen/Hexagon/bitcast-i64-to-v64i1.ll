; RUN: llc --mtriple=hexagon -mattr=+hvxv79,+hvx-length128b < %s | FileCheck %s
; CHECK-DAG: r[[REGH:([0-9]+)]]:[[REGL:([0-9]+)]] = combine(##.LCPI0_0,#-1)
; CHECK-DAG: [[VREG1:v([0-9]+)]] = vmem(r[[REGH]]+#0)
; CHECK-DAG: [[REG1:(r[0-9]+)]] = memw(r{{[0-9]+}}+#4)
; CHECK-DAG: [[VREG2:v([0-9]+)]] = vsplat([[REG1]])
; CHECK-DAG: [[REG2:(r[0-9]+)]] = memw(r{{[0-9]+}}+#0)
; CHECK-DAG: [[VREG3:v([0-9]+)]] = vsplat([[REG2]])
; CHECK-DAG: [[VREG4:v([0-9]+)]] = vand([[VREG2]],[[VREG1]])
; CHECK-DAG: [[VREG5:v([0-9]+)]] = vand([[VREG3]],[[VREG1]])
; CHECK-DAG: [[QREG:q[0-9]+]] = vand([[VREG4]],r{{[0-9]+}})
; CHECK-DAG: [[VREG6:v([0-9]+)]] = vand([[QREG]],r{{[0-9]+}})
; CHECK-DAG: [[QREG1:q[0-9]+]] = vand([[VREG5]],r{{[0-9]+}})
; CHECK-DAG: [[VREG7:v([0-9]+)]] = vand([[QREG1]],r{{[0-9]+}})
; CHECK-DAG: v{{[0-9]+}}.b = vpacke(v{{[0-9]+}}.h,v{{[0-9]+}}.h)
; CHECK-DAG: v{{[0-9]+}}.b = vpacke(v{{[0-9]+}}.h,v{{[0-9]+}}.h)
; CHECK-DAG: [[VREG8:v([0-9]+)]] = vror(v{{[0-9]+}},r{{[0-9]+}})
; CHECK-DAG: [[VREG9:v([0-9]+)]] = vor([[VREG8]],v{{[0-9]+}})
; CHECK-DAG: q{{[0-9]+}} = vand([[VREG9]],r{{[0-9]+}})
define void @bitcast_i64_to_v64i1_full(ptr %in, ptr %out) {
entry:
  %load = load i64, ptr %in, align 4
  %bitcast = bitcast i64 %load to <64 x i1>
  %e0 = extractelement <64 x i1> %bitcast, i32 0
  %e1 = extractelement <64 x i1> %bitcast, i32 1
  %z0 = zext i1 %e0 to i8
  %z1 = zext i1 %e1 to i8
  %ptr0 = getelementptr i8, ptr %out, i32 0
  %ptr1 = getelementptr i8, ptr %out, i32 1
  store i8 %z0, ptr %ptr0, align 1
  store i8 %z1, ptr %ptr1, align 1
  ret void
}

