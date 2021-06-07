; RUN: opt  -cost-model -analyze -mtriple=aarch64--linux-gnu -mattr=+sve < %s | FileCheck %s

define void @scalable_loads() {
; CHECK-LABEL: 'scalable_loads'
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction: %res.nxv8i8
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction: %res.nxv16i8
; CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction: %res.nxv32i8
  %res.nxv8i8 = load <vscale x 8 x i8>, <vscale x 8 x i8>* undef
  %res.nxv16i8 = load <vscale x 16 x i8>, <vscale x 16 x i8>* undef
  %res.nxv32i8 = load <vscale x 32 x i8>, <vscale x 32 x i8>* undef
  ret void
}

define void @scalable_stores() {
; CHECK-LABEL: 'scalable_stores'
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction: store <vscale x 8 x i8>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction: store <vscale x 16 x i8>
; CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction: store <vscale x 32 x i8>
  store <vscale x 8 x i8> undef, <vscale x 8 x i8>* undef
  store <vscale x 16 x i8> undef, <vscale x 16 x i8>* undef
  store <vscale x 32 x i8> undef, <vscale x 32 x i8>* undef
  ret void
}
