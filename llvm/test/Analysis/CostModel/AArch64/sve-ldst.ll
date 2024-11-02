; RUN: opt  -passes="print<cost-model>" 2>&1 -disable-output -mtriple=aarch64--linux-gnu -mattr=+sve < %s | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"

define void @scalable_loads() {
; CHECK-LABEL: 'scalable_loads'
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction: %res.nxv8i8
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction: %res.nxv16i8
; CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction: %res.nxv32i8
; CHECK-NEXT: Cost Model: Invalid cost for instruction: %res.nxv1i64
  %res.nxv8i8 = load <vscale x 8 x i8>, <vscale x 8 x i8>* undef
  %res.nxv16i8 = load <vscale x 16 x i8>, <vscale x 16 x i8>* undef
  %res.nxv32i8 = load <vscale x 32 x i8>, <vscale x 32 x i8>* undef
  %res.nxv1i64 = load <vscale x 1 x i64>, <vscale x 1 x i64>* undef
  ret void
}

define void @scalable_stores() {
; CHECK-LABEL: 'scalable_stores'
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction: store <vscale x 8 x i8>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction: store <vscale x 16 x i8>
; CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction: store <vscale x 32 x i8>
; CHECK-NEXT: Cost Model: Invalid cost for instruction: store <vscale x 1 x i64>
  store <vscale x 8 x i8> undef, <vscale x 8 x i8>* undef
  store <vscale x 16 x i8> undef, <vscale x 16 x i8>* undef
  store <vscale x 32 x i8> undef, <vscale x 32 x i8>* undef
  store <vscale x 1 x i64> undef, <vscale x 1 x i64>* undef
  ret void
}
