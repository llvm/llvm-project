; Check getIntrinsicInstrCost in BasicTTIImpl.h for masked gather

; RUN: opt -cost-model -analyze -mtriple=aarch64--linux-gnu -mattr=+sve  < %s | FileCheck %s

define void @masked_gathers(<vscale x 4 x i1> %nxv4i1mask, <vscale x 8 x i1> %nxv8i1mask, <4 x i1> %v4i1mask, <1 x i1> %v1i1mask, <vscale x 1 x i1> %nxv1i1mask) vscale_range(0, 16) {
; CHECK-LABEL: 'masked_gathers'
; CHECK-NEXT: Cost Model: Found an estimated cost of 64 for instruction:   %res.nxv4i32 = call <vscale x 4 x i32> @llvm.masked.gather.nxv4i32.nxv4p0i32
; CHECK-NEXT: Cost Model: Found an estimated cost of 128 for instruction:   %res.nxv8i32 = call <vscale x 8 x i32> @llvm.masked.gather.nxv8i32.nxv8p0i32
; CHECK-NEXT: Cost Model: Found an estimated cost of 29 for instruction:   %res.v4i32 = call <4 x i32> @llvm.masked.gather.v4i32.v4p0i32
; CHECK-NEXT: Cost Model: Found an estimated cost of 6 for instruction:   %res.v1i128 = call <1 x i128> @llvm.masked.gather.v1i128.v1p0i128
; CHECK-NEXT: Cost Model: Invalid cost for instruction:   %res.nxv1i64 = call <vscale x 1 x i64> @llvm.masked.gather.nxv1i64.nxv1p0i64
  %res.nxv4i32 = call <vscale x 4 x i32> @llvm.masked.gather.nxv4i32(<vscale x 4 x i32*> undef, i32 0, <vscale x 4 x i1> %nxv4i1mask, <vscale x 4 x i32> zeroinitializer)
  %res.nxv8i32 = call <vscale x 8 x i32> @llvm.masked.gather.nxv8i32(<vscale x 8 x i32*> undef, i32 0, <vscale x 8 x i1> %nxv8i1mask, <vscale x 8 x i32> zeroinitializer)
  %res.v4i32 = call <4 x i32> @llvm.masked.gather.v4i32(<4 x i32*> undef, i32 0, <4 x i1> %v4i1mask, <4 x i32> zeroinitializer)
  %res.v1i128 = call <1 x i128> @llvm.masked.gather.v1i128.v1p0i128(<1 x i128*> undef, i32 0, <1 x i1> %v1i1mask, <1 x i128> zeroinitializer)
  %res.nxv1i64 = call <vscale x 1 x i64> @llvm.masked.gather.nxv1i64.nxv1p0i64(<vscale x 1 x i64*> undef, i32 0, <vscale x 1 x i1> %nxv1i1mask, <vscale x 1 x i64> zeroinitializer)
  ret void
}

define void @masked_gathers_no_vscale_range() {
; CHECK-LABEL: 'masked_gathers_no_vscale_range'
; CHECK-NEXT:  Cost Model: Found an estimated cost of 64 for instruction: %res.nxv4f64 = call <vscale x 4 x double> @llvm.masked.gather.nxv4f64.nxv4p0f64(<vscale x 4 x double*> undef, i32 1, <vscale x 4 x i1> undef, <vscale x 4 x double> undef)
; CHECK-NEXT:  Cost Model: Found an estimated cost of 32 for instruction: %res.nxv2f64 = call <vscale x 2 x double> @llvm.masked.gather.nxv2f64.nxv2p0f64(<vscale x 2 x double*> undef, i32 1, <vscale x 2 x i1> undef, <vscale x 2 x double> undef)
; CHECK-NEXT:  Cost Model: Found an estimated cost of 128 for instruction: %res.nxv8f32 = call <vscale x 8 x float> @llvm.masked.gather.nxv8f32.nxv8p0f32(<vscale x 8 x float*> undef, i32 1, <vscale x 8 x i1> undef, <vscale x 8 x float> undef)
; CHECK-NEXT:  Cost Model: Found an estimated cost of 64 for instruction: %res.nxv4f32 = call <vscale x 4 x float> @llvm.masked.gather.nxv4f32.nxv4p0f32(<vscale x 4 x float*> undef, i32 1, <vscale x 4 x i1> undef, <vscale x 4 x float> undef)
; CHECK-NEXT:  Cost Model: Found an estimated cost of 32 for instruction: %res.nxv2f32 = call <vscale x 2 x float> @llvm.masked.gather.nxv2f32.nxv2p0f32(<vscale x 2 x float*> undef, i32 1, <vscale x 2 x i1> undef, <vscale x 2 x float> undef)
; CHECK-NEXT:  Cost Model: Found an estimated cost of 256 for instruction: %res.nxv16i16 = call <vscale x 16 x i16> @llvm.masked.gather.nxv16i16.nxv16p0i16(<vscale x 16 x i16*> undef, i32 1, <vscale x 16 x i1> undef, <vscale x 16 x i16> undef)
; CHECK-NEXT:  Cost Model: Found an estimated cost of 128 for instruction: %res.nxv8i16 = call <vscale x 8 x i16> @llvm.masked.gather.nxv8i16.nxv8p0i16(<vscale x 8 x i16*> undef, i32 1, <vscale x 8 x i1> undef, <vscale x 8 x i16> undef)
; CHECK-NEXT:  Cost Model: Found an estimated cost of 64 for instruction: %res.nxv4i16 = call <vscale x 4 x i16> @llvm.masked.gather.nxv4i16.nxv4p0i16(<vscale x 4 x i16*> undef, i32 1, <vscale x 4 x i1> undef, <vscale x 4 x i16> undef)
  %res.nxv4f64 = call <vscale x 4 x double> @llvm.masked.gather.nxv4f64(<vscale x 4 x double*> undef, i32 1, <vscale x 4 x i1> undef, <vscale x 4 x double> undef)
  %res.nxv2f64 = call <vscale x 2 x double> @llvm.masked.gather.nxv2f64(<vscale x 2 x double*> undef, i32 1, <vscale x 2 x i1> undef, <vscale x 2 x double> undef)

  %res.nxv8f32 = call <vscale x 8 x float> @llvm.masked.gather.nxv8f32(<vscale x 8 x float*> undef, i32 1, <vscale x 8 x i1> undef, <vscale x 8 x float> undef)
  %res.nxv4f32 = call <vscale x 4 x float> @llvm.masked.gather.nxv4f32(<vscale x 4 x float*> undef, i32 1, <vscale x 4 x i1> undef, <vscale x 4 x float> undef)
  %res.nxv2f32 = call <vscale x 2 x float> @llvm.masked.gather.nxv2f32(<vscale x 2 x float*> undef, i32 1, <vscale x 2 x i1> undef, <vscale x 2 x float> undef)

  %res.nxv16i16 = call <vscale x 16 x i16> @llvm.masked.gather.nxv16i16(<vscale x 16 x i16*> undef, i32 1, <vscale x 16 x i1> undef, <vscale x 16 x i16> undef)
  %res.nxv8i16  = call <vscale x 8 x i16>  @llvm.masked.gather.nxv8i16(<vscale x 8 x i16*> undef, i32 1, <vscale x 8 x i1> undef, <vscale x 8 x i16> undef)
  %res.nxv4i16  = call <vscale x 4 x i16>  @llvm.masked.gather.nxv4i16(<vscale x 4 x i16*> undef, i32 1, <vscale x 4 x i1> undef, <vscale x 4 x i16> undef)

  ret void
}

declare <vscale x 4 x i32> @llvm.masked.gather.nxv4i32(<vscale x 4 x i32*>, i32, <vscale x 4 x i1>, <vscale x 4 x i32>)
declare <vscale x 8 x i32> @llvm.masked.gather.nxv8i32(<vscale x 8 x i32*>, i32, <vscale x 8 x i1>, <vscale x 8 x i32>)
declare <4 x i32> @llvm.masked.gather.v4i32(<4 x i32*>, i32, <4 x i1>, <4 x i32>)
declare <1 x i128> @llvm.masked.gather.v1i128.v1p0i128(<1 x i128*>, i32, <1 x i1>, <1 x i128>)
declare <vscale x 1 x i64> @llvm.masked.gather.nxv1i64.nxv1p0i64(<vscale x 1 x i64*>, i32, <vscale x 1 x i1>, <vscale x 1 x i64>)
declare <vscale x 4 x double> @llvm.masked.gather.nxv4f64(<vscale x 4 x double*>, i32, <vscale x 4 x i1>, <vscale x 4 x double>)
declare <vscale x 2 x double> @llvm.masked.gather.nxv2f64(<vscale x 2 x double*>, i32, <vscale x 2 x i1>, <vscale x 2 x double>)
declare <vscale x 8 x float> @llvm.masked.gather.nxv8f32(<vscale x 8 x float*>, i32, <vscale x 8 x i1>, <vscale x 8 x float>)
declare <vscale x 4 x float> @llvm.masked.gather.nxv4f32(<vscale x 4 x float*>, i32, <vscale x 4 x i1>, <vscale x 4 x float>)
declare <vscale x 2 x float> @llvm.masked.gather.nxv2f32(<vscale x 2 x float*>, i32, <vscale x 2 x i1>, <vscale x 2 x float>)
declare <vscale x 16 x i16> @llvm.masked.gather.nxv16i16(<vscale x 16 x i16*>, i32, <vscale x 16 x i1>, <vscale x 16 x i16>)
declare <vscale x 8 x i16> @llvm.masked.gather.nxv8i16(<vscale x 8 x i16*>, i32, <vscale x 8 x i1>, <vscale x 8 x i16>)
declare <vscale x 4 x i16> @llvm.masked.gather.nxv4i16(<vscale x 4 x i16*>, i32, <vscale x 4 x i1>, <vscale x 4 x i16>)
