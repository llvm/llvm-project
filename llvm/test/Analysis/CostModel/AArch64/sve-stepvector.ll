; RUN: opt -passes="print<cost-model>" 2>&1 -disable-output -mtriple=aarch64--linux-gnu -mattr=+sve  < %s | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"

; Check icmp for legal integer vectors.
define void @stepvector_legal_int() {
; CHECK-LABEL: 'stepvector_legal_int'
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %1 = call <vscale x 2 x i64> @llvm.stepvector.nxv2i64()
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %2 = call <vscale x 4 x i32> @llvm.stepvector.nxv4i32()
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %3 = call <vscale x 8 x i16> @llvm.stepvector.nxv8i16()
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %4 = call <vscale x 16 x i8> @llvm.stepvector.nxv16i8()
  %1 = call <vscale x 2 x i64> @llvm.stepvector.nxv2i64()
  %2 = call <vscale x 4 x i32> @llvm.stepvector.nxv4i32()
  %3 = call <vscale x 8 x i16> @llvm.stepvector.nxv8i16()
  %4 = call <vscale x 16 x i8> @llvm.stepvector.nxv16i8()
  ret void
}

; Check icmp for an illegal integer vector.
define void @stepvector_illegal_int() {
; CHECK-LABEL: 'stepvector_illegal_int'
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %1 = call <vscale x 4 x i64> @llvm.stepvector.nxv4i64()
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %2 = call <vscale x 16 x i32> @llvm.stepvector.nxv16i32()
  %1 = call <vscale x 4 x i64> @llvm.stepvector.nxv4i64()
  %2 = call <vscale x 16 x i32> @llvm.stepvector.nxv16i32()
  ret void
}


declare <vscale x 2 x i64> @llvm.stepvector.nxv2i64()
declare <vscale x 4 x i32> @llvm.stepvector.nxv4i32()
declare <vscale x 8 x i16> @llvm.stepvector.nxv8i16()
declare <vscale x 16 x i8> @llvm.stepvector.nxv16i8()

declare <vscale x 4 x i64> @llvm.stepvector.nxv4i64()
declare <vscale x 16 x i32> @llvm.stepvector.nxv16i32()
