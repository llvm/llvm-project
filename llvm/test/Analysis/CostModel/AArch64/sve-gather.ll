; Check getIntrinsicInstrCost in BasicTTIImpl.h for masked gather

; RUN: opt -cost-model -analyze -mtriple=aarch64--linux-gnu -mattr=+sve  < %s | FileCheck %s

define void @masked_gathers(<vscale x 4 x i1> %nxv4i1mask, <vscale x 8 x i1> %nxv8i1mask, <4 x i1> %v4i1mask, <1 x i1> %v1i1mask, <vscale x 1 x i1> %nxv1i1mask) {
; CHECK-LABEL: 'masked_gathers'
; CHECK-NEXT: Cost Model: Found an estimated cost of 64 for instruction:   %res.nxv4i32 = call <vscale x 4 x i32> @llvm.masked.gather.nxv4i32.nxv4p0i32
; CHECK-NEXT: Cost Model: Found an estimated cost of 128 for instruction:   %res.nxv8i32 = call <vscale x 8 x i32> @llvm.masked.gather.nxv8i32.nxv8p0i32
; CHECK-NEXT: Cost Model: Found an estimated cost of 29 for instruction:   %res.v4i32 = call <4 x i32> @llvm.masked.gather.v4i32.v4p0i32
; CHECK-NEXT: Cost Model: Found an estimated cost of 6 for instruction:   %res.v1i128 = call <1 x i128> @llvm.masked.gather.v1i128.v1p0i128
  %res.nxv4i32 = call <vscale x 4 x i32> @llvm.masked.gather.nxv4i32(<vscale x 4 x i32*> undef, i32 0, <vscale x 4 x i1> %nxv4i1mask, <vscale x 4 x i32> zeroinitializer)
  %res.nxv8i32 = call <vscale x 8 x i32> @llvm.masked.gather.nxv8i32(<vscale x 8 x i32*> undef, i32 0, <vscale x 8 x i1> %nxv8i1mask, <vscale x 8 x i32> zeroinitializer)
  %res.v4i32 = call <4 x i32> @llvm.masked.gather.v4i32(<4 x i32*> undef, i32 0, <4 x i1> %v4i1mask, <4 x i32> zeroinitializer)
  %res.v1i128 = call <1 x i128> @llvm.masked.gather.v1i128.v1p0i128(<1 x i128*> undef, i32 0, <1 x i1> %v1i1mask, <1 x i128> zeroinitializer)
  ret void
}

declare <vscale x 4 x i32> @llvm.masked.gather.nxv4i32(<vscale x 4 x i32*>, i32, <vscale x 4 x i1>, <vscale x 4 x i32>)
declare <vscale x 8 x i32> @llvm.masked.gather.nxv8i32(<vscale x 8 x i32*>, i32, <vscale x 8 x i1>, <vscale x 8 x i32>)
declare <4 x i32> @llvm.masked.gather.v4i32(<4 x i32*>, i32, <4 x i1>, <4 x i32>)
declare <1 x i128> @llvm.masked.gather.v1i128.v1p0i128(<1 x i128*>, i32, <1 x i1>, <1 x i128>)
