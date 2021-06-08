; Check getIntrinsicInstrCost in BasicTTIImpl.h with for  masked scatter

; RUN: opt -cost-model -analyze -mtriple=aarch64--linux-gnu -mattr=+sve  < %s | FileCheck %s

define void @masked_scatters(<vscale x 4 x i1> %nxv4i1mask, <vscale x 8 x i1> %nxv8i1mask, <4 x i1> %v4i1mask, <1 x i1> %v1i1mask, <vscale x 1 x i1> %nxv1i1mask) {
; CHECK-LABEL: 'masked_scatters'
; CHECK-NEXT: Cost Model: Found an estimated cost of 64 for instruction: call void @llvm.masked.scatter.nxv4i32.nxv4p0i32
; CHECK-NEXT: Cost Model: Found an estimated cost of 128 for instruction: call void @llvm.masked.scatter.nxv8i32.nxv8p0i32
; CHECK-NEXT: Cost Model: Found an estimated cost of 29 for instruction: call void @llvm.masked.scatter.v4i32.v4p0i32
; CHECK-NEXT: Cost Model: Found an estimated cost of 6 for instruction: call void @llvm.masked.scatter.v1i128.v1p0i128
  call void @llvm.masked.scatter.nxv4i32(<vscale x 4 x i32> undef, <vscale x 4 x i32*> undef, i32 0, <vscale x 4 x i1> %nxv4i1mask)
  call void @llvm.masked.scatter.nxv8i32(<vscale x 8 x i32> undef, <vscale x 8 x i32*> undef, i32 0, <vscale x 8 x i1> %nxv8i1mask)
  call void @llvm.masked.scatter.v4i32(<4 x i32> undef, <4 x i32*> undef, i32 0, <4 x i1> %v4i1mask)
  call void @llvm.masked.scatter.v1i128.v1p0i128(<1 x i128> undef, <1 x i128*> undef, i32 0, <1 x i1> %v1i1mask)
  ret void
}

declare void @llvm.masked.scatter.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32*>, i32, <vscale x 4 x i1>)
declare void @llvm.masked.scatter.nxv8i32(<vscale x 8 x i32>, <vscale x 8 x i32*>, i32, <vscale x 8 x i1>)
declare void @llvm.masked.scatter.v4i32(<4 x i32>, <4 x i32*>, i32, <4 x i1>)
declare void @llvm.masked.scatter.v1i128.v1p0i128(<1 x i128>, <1 x i128*>, i32, <1 x i1>)
