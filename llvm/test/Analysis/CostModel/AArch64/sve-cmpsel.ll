; RUN: opt -passes='print<cost-model>' 2>&1 -disable-output -mtriple=aarch64--linux-gnu -mattr=+sve  < %s | FileCheck %s

; Check icmp for legal integer vectors.
define void @cmp_legal_int() {
; CHECK-LABEL: 'cmp_legal_int'
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %1 = icmp ne <vscale x 2 x i64> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %2 = icmp ne <vscale x 4 x i32> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %3 = icmp ne <vscale x 8 x i16> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %4 = icmp ne <vscale x 16 x i8> undef, undef
  %1 = icmp ne <vscale x 2 x i64> undef, undef
  %2 = icmp ne <vscale x 4 x i32> undef, undef
  %3 = icmp ne <vscale x 8 x i16> undef, undef
  %4 = icmp ne <vscale x 16 x i8> undef, undef
  ret void
}

; Check icmp for an illegal integer vector.
define <vscale x 4 x i1> @cmp_nxv4i64() {
; CHECK-LABEL: 'cmp_nxv4i64'
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %res = icmp ne <vscale x 4 x i64> undef, undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   ret <vscale x 4 x i1> %res
  %res = icmp ne <vscale x 4 x i64> undef, undef
  ret <vscale x 4 x i1> %res
}

; Check icmp for legal predicate vectors.
define void @cmp_legal_pred() {
; CHECK-LABEL: 'cmp_legal_pred'
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %1 = icmp ne <vscale x 2 x i1> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %2 = icmp ne <vscale x 4 x i1> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %3 = icmp ne <vscale x 8 x i1> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %4 = icmp ne <vscale x 16 x i1> undef, undef
  %1 = icmp ne <vscale x 2 x i1> undef, undef
  %2 = icmp ne <vscale x 4 x i1> undef, undef
  %3 = icmp ne <vscale x 8 x i1> undef, undef
  %4 = icmp ne <vscale x 16 x i1> undef, undef
  ret void
}

; Check icmp for an illegal predicate vector.
define <vscale x 32 x i1> @cmp_nxv32i1() {
; CHECK-LABEL: 'cmp_nxv32i1'
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %res = icmp ne <vscale x 32 x i1> undef, undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   ret <vscale x 32 x i1> %res
  %res = icmp ne <vscale x 32 x i1> undef, undef
  ret <vscale x 32 x i1> %res
}

; Check fcmp for legal FP vectors
define void @cmp_legal_fp() #0 {
; CHECK-LABEL: 'cmp_legal_fp'
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %1 = fcmp oge <vscale x 2 x double> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %2 = fcmp oge <vscale x 4 x float> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %3 = fcmp oge <vscale x 8 x half> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %4 = fcmp oge <vscale x 8 x bfloat> undef, undef
  %1 = fcmp oge <vscale x 2 x double> undef, undef
  %2 = fcmp oge <vscale x 4 x float> undef, undef
  %3 = fcmp oge <vscale x 8 x half> undef, undef
  %4 = fcmp oge <vscale x 8 x bfloat> undef, undef
  ret void
}

; Check fcmp for an illegal FP vector
define <vscale x 16 x i1> @cmp_nxv16f16() {
; CHECK-LABEL: 'cmp_nxv16f16'
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %res = fcmp oge <vscale x 16 x half> undef, undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   ret <vscale x 16 x i1> %res
  %res = fcmp oge <vscale x 16 x half> undef, undef
  ret <vscale x 16 x i1> %res
}

; Check select for legal integer vectors
define void @sel_legal_int() {
; CHECK-LABEL: 'sel_legal_int'
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %1 = select <vscale x 2 x i1> undef, <vscale x 2 x i64> undef, <vscale x 2 x i64> undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %2 = select <vscale x 4 x i1> undef, <vscale x 4 x i32> undef, <vscale x 4 x i32> undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %3 = select <vscale x 8 x i1> undef, <vscale x 8 x i16> undef, <vscale x 8 x i16> undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %4 = select <vscale x 16 x i1> undef, <vscale x 16 x i8> undef, <vscale x 16 x i8> undef
  %1 = select <vscale x 2 x i1> undef, <vscale x 2 x i64> undef, <vscale x 2 x i64> undef
  %2 = select <vscale x 4 x i1> undef, <vscale x 4 x i32> undef, <vscale x 4 x i32> undef
  %3 = select <vscale x 8 x i1> undef, <vscale x 8 x i16> undef, <vscale x 8 x i16> undef
  %4 = select <vscale x 16 x i1> undef, <vscale x 16 x i8> undef, <vscale x 16 x i8> undef
  ret void
}

; Check select for an illegal integer vector
define <vscale x 16 x i16> @sel_nxv16i16() {
; CHECK-LABEL: 'sel_nxv16i16'
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %res = select <vscale x 16 x i1> undef, <vscale x 16 x i16> undef, <vscale x 16 x i16> undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   ret <vscale x 16 x i16> %res
  %res = select <vscale x 16 x i1> undef, <vscale x 16 x i16> undef, <vscale x 16 x i16> undef
  ret <vscale x 16 x i16> %res
}

; Check select for a legal FP vector
define void @sel_legal_fp() #0 {
; CHECK-LABEL: 'sel_legal_fp'
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %1 = select <vscale x 2 x i1> undef, <vscale x 2 x double> undef, <vscale x 2 x double> undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %2 = select <vscale x 4 x i1> undef, <vscale x 4 x float> undef, <vscale x 4 x float> undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %3 = select <vscale x 8 x i1> undef, <vscale x 8 x half> undef, <vscale x 8 x half> undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %4 = select <vscale x 8 x i1> undef, <vscale x 8 x bfloat> undef, <vscale x 8 x bfloat> undef
  %1 = select <vscale x 2 x i1> undef, <vscale x 2 x double> undef, <vscale x 2 x double> undef
  %2 = select <vscale x 4 x i1> undef, <vscale x 4 x float> undef, <vscale x 4 x float> undef
  %3 = select <vscale x 8 x i1> undef, <vscale x 8 x half> undef, <vscale x 8 x half> undef
  %4 = select <vscale x 8 x i1> undef, <vscale x 8 x bfloat> undef, <vscale x 8 x bfloat> undef
  ret void
}

; Check select for an illegal FP vector
define <vscale x 8 x float> @sel_nxv8f32() {
; CHECK-LABEL: 'sel_nxv8f32'
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %res = select <vscale x 8 x i1> undef, <vscale x 8 x float> undef, <vscale x 8 x float> undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   ret <vscale x 8 x float> %res
  %res = select <vscale x 8 x i1> undef, <vscale x 8 x float> undef, <vscale x 8 x float> undef
  ret <vscale x 8 x float> %res
}

; Check select for a legal predicate vector
define void @sel_legal_pred() {
; CHECK-LABEL: 'sel_legal_pred'
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %1 = select <vscale x 2 x i1> undef, <vscale x 2 x i1> undef, <vscale x 2 x i1> undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %2 = select <vscale x 4 x i1> undef, <vscale x 4 x i1> undef, <vscale x 4 x i1> undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %3 = select <vscale x 8 x i1> undef, <vscale x 8 x i1> undef, <vscale x 8 x i1> undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %4 = select <vscale x 16 x i1> undef, <vscale x 16 x i1> undef, <vscale x 16 x i1> undef
  %1 = select <vscale x 2 x i1> undef, <vscale x 2 x i1> undef, <vscale x 2 x i1> undef
  %2 = select <vscale x 4 x i1> undef, <vscale x 4 x i1> undef, <vscale x 4 x i1> undef
  %3 = select <vscale x 8 x i1> undef, <vscale x 8 x i1> undef, <vscale x 8 x i1> undef
  %4 = select <vscale x 16 x i1> undef, <vscale x 16 x i1> undef, <vscale x 16 x i1> undef
  ret void
}

; Check select for an illegal predicate vector
define <vscale x 32 x i1> @sel_nxv32i1() {
; CHECK-LABEL: 'sel_nxv32i1'
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %res = select <vscale x 32 x i1> undef, <vscale x 32 x i1> undef, <vscale x 32 x i1> undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   ret <vscale x 32 x i1> %res
  %res = select <vscale x 32 x i1> undef, <vscale x 32 x i1> undef, <vscale x 32 x i1> undef
  ret <vscale x 32 x i1> %res
}

attributes #0 = { "target-features"="+sve,+bf16" }
