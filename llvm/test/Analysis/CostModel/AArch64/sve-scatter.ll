; Check getIntrinsicInstrCost in BasicTTIImpl.h with for  masked scatter

; RUN: opt -cost-model -analyze -mtriple=aarch64--linux-gnu -mattr=+sve  < %s | FileCheck %s

define void @masked_scatters(<vscale x 4 x i1> %nxv4i1mask, <vscale x 8 x i1> %nxv8i1mask, <4 x i1> %v4i1mask, <1 x i1> %v1i1mask, <vscale x 1 x i1> %nxv1i1mask) vscale_range(0, 16) {
; CHECK-LABEL: 'masked_scatters'
; CHECK-NEXT: Cost Model: Found an estimated cost of 64 for instruction: call void @llvm.masked.scatter.nxv4i32.nxv4p0i32
; CHECK-NEXT: Cost Model: Found an estimated cost of 128 for instruction: call void @llvm.masked.scatter.nxv8i32.nxv8p0i32
; CHECK-NEXT: Cost Model: Found an estimated cost of 29 for instruction: call void @llvm.masked.scatter.v4i32.v4p0i32
; CHECK-NEXT: Cost Model: Found an estimated cost of 6 for instruction: call void @llvm.masked.scatter.v1i128.v1p0i128
; CHECK-NEXT: Cost Model: Invalid cost for instruction: call void @llvm.masked.scatter.nxv1i64.nxv1p0i64
  call void @llvm.masked.scatter.nxv4i32(<vscale x 4 x i32> undef, <vscale x 4 x i32*> undef, i32 0, <vscale x 4 x i1> %nxv4i1mask)
  call void @llvm.masked.scatter.nxv8i32(<vscale x 8 x i32> undef, <vscale x 8 x i32*> undef, i32 0, <vscale x 8 x i1> %nxv8i1mask)
  call void @llvm.masked.scatter.v4i32(<4 x i32> undef, <4 x i32*> undef, i32 0, <4 x i1> %v4i1mask)
  call void @llvm.masked.scatter.v1i128.v1p0i128(<1 x i128> undef, <1 x i128*> undef, i32 0, <1 x i1> %v1i1mask)
  call void @llvm.masked.scatter.nxv1i64.nxv1p0i64(<vscale x 1 x i64> undef, <vscale x 1 x i64*> undef, i32 0, <vscale x 1 x i1> %nxv1i1mask)
  ret void
}

define void @masked_scatters_no_vscale_range() {
; CHECK-LABEL: 'masked_scatters_no_vscale_range'
; CHECK-NEXT:  Cost Model: Found an estimated cost of 64 for instruction: call void @llvm.masked.scatter.nxv4f64.nxv4p0f64(<vscale x 4 x double> undef, <vscale x 4 x double*> undef, i32 1, <vscale x 4 x i1> undef)
; CHECK-NEXT:  Cost Model: Found an estimated cost of 32 for instruction: call void @llvm.masked.scatter.nxv2f64.nxv2p0f64(<vscale x 2 x double> undef, <vscale x 2 x double*> undef, i32 1, <vscale x 2 x i1> undef)
; CHECK-NEXT:  Cost Model: Found an estimated cost of 128 for instruction: call void @llvm.masked.scatter.nxv8f32.nxv8p0f32(<vscale x 8 x float> undef, <vscale x 8 x float*> undef, i32 1, <vscale x 8 x i1> undef)
; CHECK-NEXT:  Cost Model: Found an estimated cost of 64 for instruction: call void @llvm.masked.scatter.nxv4f32.nxv4p0f32(<vscale x 4 x float> undef, <vscale x 4 x float*> undef, i32 1, <vscale x 4 x i1> undef)
; CHECK-NEXT:  Cost Model: Found an estimated cost of 32 for instruction: call void @llvm.masked.scatter.nxv2f32.nxv2p0f32(<vscale x 2 x float> undef, <vscale x 2 x float*> undef, i32 1, <vscale x 2 x i1> undef)
; CHECK-NEXT:  Cost Model: Found an estimated cost of 256 for instruction: call void @llvm.masked.scatter.nxv16i16.nxv16p0i16(<vscale x 16 x i16> undef, <vscale x 16 x i16*> undef, i32 1, <vscale x 16 x i1> undef)
; CHECK-NEXT:  Cost Model: Found an estimated cost of 128 for instruction: call void @llvm.masked.scatter.nxv8i16.nxv8p0i16(<vscale x 8 x i16> undef, <vscale x 8 x i16*> undef, i32 1, <vscale x 8 x i1> undef)
; CHECK-NEXT:  Cost Model: Found an estimated cost of 64 for instruction: call void @llvm.masked.scatter.nxv4i16.nxv4p0i16(<vscale x 4 x i16> undef, <vscale x 4 x i16*> undef, i32 1, <vscale x 4 x i1> undef)
  call void @llvm.masked.scatter.nxv4f64(<vscale x 4 x double> undef, <vscale x 4 x double*> undef, i32 1, <vscale x 4 x i1> undef)
  call void @llvm.masked.scatter.nxv2f64(<vscale x 2 x double> undef, <vscale x 2 x double*> undef, i32 1, <vscale x 2 x i1> undef)

  call void @llvm.masked.scatter.nxv8f32(<vscale x 8 x float> undef, <vscale x 8 x float*> undef, i32 1, <vscale x 8 x i1> undef)
  call void @llvm.masked.scatter.nxv4f32(<vscale x 4 x float> undef, <vscale x 4 x float*> undef, i32 1, <vscale x 4 x i1> undef)
  call void @llvm.masked.scatter.nxv2f32(<vscale x 2 x float> undef, <vscale x 2 x float*> undef, i32 1, <vscale x 2 x i1> undef)

  call void @llvm.masked.scatter.nxv16i16(<vscale x 16 x i16> undef, <vscale x 16 x i16*> undef, i32 1, <vscale x 16 x i1> undef)
  call void @llvm.masked.scatter.nxv8i16(<vscale x 8 x i16> undef, <vscale x 8 x i16*> undef, i32 1, <vscale x 8 x i1> undef)
  call void @llvm.masked.scatter.nxv4i16(<vscale x 4 x i16> undef, <vscale x 4 x i16*> undef, i32 1, <vscale x 4 x i1> undef)

  ret void
}

declare void @llvm.masked.scatter.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32*>, i32, <vscale x 4 x i1>)
declare void @llvm.masked.scatter.nxv8i32(<vscale x 8 x i32>, <vscale x 8 x i32*>, i32, <vscale x 8 x i1>)
declare void @llvm.masked.scatter.v4i32(<4 x i32>, <4 x i32*>, i32, <4 x i1>)
declare void @llvm.masked.scatter.v1i128.v1p0i128(<1 x i128>, <1 x i128*>, i32, <1 x i1>)
declare void @llvm.masked.scatter.nxv1i64.nxv1p0i64(<vscale x 1 x i64>, <vscale x 1 x i64*>, i32, <vscale x 1 x i1>)
declare void @llvm.masked.scatter.nxv4f64(<vscale x 4 x double>, <vscale x 4 x double*>, i32, <vscale x 4 x i1>)
declare void @llvm.masked.scatter.nxv2f64(<vscale x 2 x double>, <vscale x 2 x double*>, i32, <vscale x 2 x i1>)
declare void @llvm.masked.scatter.nxv8f32(<vscale x 8 x float>, <vscale x 8 x float*>, i32, <vscale x 8 x i1>)
declare void @llvm.masked.scatter.nxv4f32(<vscale x 4 x float>, <vscale x 4 x float*>, i32, <vscale x 4 x i1>)
declare void @llvm.masked.scatter.nxv2f32(<vscale x 2 x float>, <vscale x 2 x float*>, i32, <vscale x 2 x i1>)
declare void @llvm.masked.scatter.nxv16i16(<vscale x 16 x i16>, <vscale x 16 x i16*>, i32, <vscale x 16 x i1>)
declare void @llvm.masked.scatter.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16*>, i32, <vscale x 8 x i1>)
declare void @llvm.masked.scatter.nxv4i16(<vscale x 4 x i16>, <vscale x 4 x i16*>, i32, <vscale x 4 x i1>)
