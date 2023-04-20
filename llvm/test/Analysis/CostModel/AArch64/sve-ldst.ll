; RUN: opt  -passes="print<cost-model>" 2>&1 -disable-output -mtriple=aarch64--linux-gnu -mattr=+sve < %s | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"

define void @scalable_loads() {
; CHECK-LABEL: 'scalable_loads'
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction: %res.nxv8i8
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction: %res.nxv16i8
; CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction: %res.nxv32i8
; CHECK-NEXT: Cost Model: Invalid cost for instruction: %res.nxv1i64
  %res.nxv8i8 = load <vscale x 8 x i8>, ptr undef
  %res.nxv16i8 = load <vscale x 16 x i8>, ptr undef
  %res.nxv32i8 = load <vscale x 32 x i8>, ptr undef
  %res.nxv1i64 = load <vscale x 1 x i64>, ptr undef
  ret void
}

define void @scalable_stores() {
; CHECK-LABEL: 'scalable_stores'
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction: store <vscale x 8 x i8>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction: store <vscale x 16 x i8>
; CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction: store <vscale x 32 x i8>
; CHECK-NEXT: Cost Model: Invalid cost for instruction: store <vscale x 1 x i64>
  store <vscale x 8 x i8> undef, ptr undef
  store <vscale x 16 x i8> undef, ptr undef
  store <vscale x 32 x i8> undef, ptr undef
  store <vscale x 1 x i64> undef, ptr undef
  ret void
}

define void @scalable_ext_loads() {
; CHECK-LABEL: 'scalable_ext_loads'
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %load.nxv16i8 = load <vscale x 16 x i8>, ptr undef, align 16
; CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %zext.nxv16i8to16 = zext <vscale x 16 x i8> %load.nxv16i8 to <vscale x 16 x i16>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %load.nxv8i8 = load <vscale x 8 x i8>, ptr undef, align 8
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   %zext.nxv8i8to16 = zext <vscale x 8 x i8> %load.nxv8i8 to <vscale x 8 x i16>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %load.nxv4i8 = load <vscale x 4 x i8>, ptr undef, align 4
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   %zext.nxv4i8to32 = zext <vscale x 4 x i8> %load.nxv4i8 to <vscale x 4 x i32>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %load.nxv2i8 = load <vscale x 2 x i8>, ptr undef, align 2
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   %zext.nxv2i8to64 = zext <vscale x 2 x i8> %load.nxv2i8 to <vscale x 2 x i64>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %load.nxv8i16 = load <vscale x 8 x i16>, ptr undef, align 16
; CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %zext.nxv8i16to32 = zext <vscale x 8 x i16> %load.nxv8i16 to <vscale x 8 x i32>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %load.nxv4i16 = load <vscale x 4 x i16>, ptr undef, align 8
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   %zext.nxv4i16to32 = zext <vscale x 4 x i16> %load.nxv4i16 to <vscale x 4 x i32>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %load.nxv2i16 = load <vscale x 2 x i16>, ptr undef, align 4
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   %zext.nxv2i16to64 = zext <vscale x 2 x i16> %load.nxv2i16 to <vscale x 2 x i64>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %load.nxv4i32 = load <vscale x 4 x i32>, ptr undef, align 16
; CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %zext.nxv4i32to64 = zext <vscale x 4 x i32> %load.nxv4i32 to <vscale x 4 x i64>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %load.nxv2i32 = load <vscale x 2 x i32>, ptr undef, align 8
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   %zext.nxv2i32to64 = zext <vscale x 2 x i32> %load.nxv2i32 to <vscale x 2 x i64>

; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %load2.nxv16i8 = load <vscale x 16 x i8>, ptr undef, align 16
; CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %sext.nxv16i8to16 = sext <vscale x 16 x i8> %load2.nxv16i8 to <vscale x 16 x i16>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %load2.nxv8i8 = load <vscale x 8 x i8>, ptr undef, align 8
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   %sext.nxv8i8to16 = sext <vscale x 8 x i8> %load2.nxv8i8 to <vscale x 8 x i16>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %load2.nxv4i8 = load <vscale x 4 x i8>, ptr undef, align 4
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   %sext.nxv4i8to32 = sext <vscale x 4 x i8> %load2.nxv4i8 to <vscale x 4 x i32>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %load2.nxv2i8 = load <vscale x 2 x i8>, ptr undef, align 2
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   %sext.nxv2i8to64 = sext <vscale x 2 x i8> %load2.nxv2i8 to <vscale x 2 x i64>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %load2.nxv8i16 = load <vscale x 8 x i16>, ptr undef, align 16
; CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %sext.nxv8i16to32 = sext <vscale x 8 x i16> %load2.nxv8i16 to <vscale x 8 x i32>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %load2.nxv4i16 = load <vscale x 4 x i16>, ptr undef, align 8
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   %sext.nxv4i16to32 = sext <vscale x 4 x i16> %load2.nxv4i16 to <vscale x 4 x i32>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %load2.nxv2i16 = load <vscale x 2 x i16>, ptr undef, align 4
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   %sext.nxv2i16to64 = sext <vscale x 2 x i16> %load2.nxv2i16 to <vscale x 2 x i64>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %load2.nxv4i32 = load <vscale x 4 x i32>, ptr undef, align 16
; CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %sext.nxv4i32to64 = sext <vscale x 4 x i32> %load2.nxv4i32 to <vscale x 4 x i64>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %load2.nxv2i32 = load <vscale x 2 x i32>, ptr undef, align 8
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   %sext.nxv2i32to64 = sext <vscale x 2 x i32> %load2.nxv2i32 to <vscale x 2 x i64>

  %load.nxv16i8 = load <vscale x 16 x i8>, ptr undef
  %zext.nxv16i8to16 = zext <vscale x 16 x i8> %load.nxv16i8 to <vscale x 16 x i16>
  %load.nxv8i8 = load <vscale x 8 x i8>, ptr undef
  %zext.nxv8i8to16 = zext <vscale x 8 x i8> %load.nxv8i8 to <vscale x 8 x i16>
  %load.nxv4i8 = load <vscale x 4 x i8>, ptr undef
  %zext.nxv4i8to32 = zext <vscale x 4 x i8> %load.nxv4i8 to <vscale x 4 x i32>
  %load.nxv2i8 = load <vscale x 2 x i8>, ptr undef
  %zext.nxv2i8to64 = zext <vscale x 2 x i8> %load.nxv2i8 to <vscale x 2 x i64>
  %load.nxv8i16 = load <vscale x 8 x i16>, ptr undef
  %zext.nxv8i16to32 = zext <vscale x 8 x i16> %load.nxv8i16 to <vscale x 8 x i32>
  %load.nxv4i16 = load <vscale x 4 x i16>, ptr undef
  %zext.nxv4i16to32 = zext <vscale x 4 x i16> %load.nxv4i16 to <vscale x 4 x i32>
  %load.nxv2i16 = load <vscale x 2 x i16>, ptr undef
  %zext.nxv2i16to64 = zext <vscale x 2 x i16> %load.nxv2i16 to <vscale x 2 x i64>
  %load.nxv4i32 = load <vscale x 4 x i32>, ptr undef
  %zext.nxv4i32to64 = zext <vscale x 4 x i32> %load.nxv4i32 to <vscale x 4 x i64>
  %load.nxv2i32 = load <vscale x 2 x i32>, ptr undef
  %zext.nxv2i32to64 = zext <vscale x 2 x i32> %load.nxv2i32 to <vscale x 2 x i64>

  %load2.nxv16i8 = load <vscale x 16 x i8>, ptr undef
  %sext.nxv16i8to16 = sext <vscale x 16 x i8> %load2.nxv16i8 to <vscale x 16 x i16>
  %load2.nxv8i8 = load <vscale x 8 x i8>, ptr undef
  %sext.nxv8i8to16 = sext <vscale x 8 x i8> %load2.nxv8i8 to <vscale x 8 x i16>
  %load2.nxv4i8 = load <vscale x 4 x i8>, ptr undef
  %sext.nxv4i8to32 = sext <vscale x 4 x i8> %load2.nxv4i8 to <vscale x 4 x i32>
  %load2.nxv2i8 = load <vscale x 2 x i8>, ptr undef
  %sext.nxv2i8to64 = sext <vscale x 2 x i8> %load2.nxv2i8 to <vscale x 2 x i64>
  %load2.nxv8i16 = load <vscale x 8 x i16>, ptr undef
  %sext.nxv8i16to32 = sext <vscale x 8 x i16> %load2.nxv8i16 to <vscale x 8 x i32>
  %load2.nxv4i16 = load <vscale x 4 x i16>, ptr undef
  %sext.nxv4i16to32 = sext <vscale x 4 x i16> %load2.nxv4i16 to <vscale x 4 x i32>
  %load2.nxv2i16 = load <vscale x 2 x i16>, ptr undef
  %sext.nxv2i16to64 = sext <vscale x 2 x i16> %load2.nxv2i16 to <vscale x 2 x i64>
  %load2.nxv4i32 = load <vscale x 4 x i32>, ptr undef
  %sext.nxv4i32to64 = sext <vscale x 4 x i32> %load2.nxv4i32 to <vscale x 4 x i64>
  %load2.nxv2i32 = load <vscale x 2 x i32>, ptr undef
  %sext.nxv2i32to64 = sext <vscale x 2 x i32> %load2.nxv2i32 to <vscale x 2 x i64>

  ret void
}

;; NOTE: These prefixes are unused and the list is autogenerated. Do not add tests below this line:
; CHECK: {{.*}}
