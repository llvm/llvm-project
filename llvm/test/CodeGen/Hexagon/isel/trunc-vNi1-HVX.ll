; RUN: llc --mtriple=hexagon -mattr=+hvxv79,+hvx-length128b < %s | FileCheck %s

define void @f5(<64 x i32> %a0, ptr %a1) {
; CHECK-LABEL: f5:
; CHECK: [[REG0:(r[0-9]+)]] = ##16843009
; CHECK-DAG: q[[Q0:[0-9]+]] = vand(v{{[0-9]+}},[[REG0]])
; CHECK-DAG: q[[Q1:[0-9]+]] = vand(v{{[0-9]+}},[[REG0]])
; CHECK: v{{[0-9]+}}.b = vpacke(v{{[0-9]+}}.h,v{{[0-9]+}}.h)
; CHECK: v{{[0-9]+}}.b = vpacke(v{{[0-9]+}}.h,v{{[0-9]+}}.h)
; CHECK: v[[VROR:[0-9]+]] = vror(v{{[0-9]+}},r{{[0-9]+}})
; CHECK: v[[VOR:[0-9]+]] = vor(v[[VROR]],v{{[0-9]+}})
; CHECK: q{{[0-9]+}} = vand(v[[VOR]],r{{[0-9]+}})
b0:
  %v0 = trunc <64 x i32> %a0 to <64 x i1>
  store <64 x i1> %v0, ptr %a1, align 1
  ret void
}

