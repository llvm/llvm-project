; RUN: opt < %s -cost-model -analyze -mtriple=systemz-unknown -mcpu=z13 | FileCheck %s

define void @bswap_i64(i64 %arg, <2 x i64> %arg2) {
; CHECK: Printing analysis 'Cost Model Analysis' for function 'bswap_i64':
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %swp1 = tail call i64
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %swp2 = tail call <2 x i64>
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %swp4 = tail call <4 x i64>
  %swp1 = tail call i64 @llvm.bswap.i64(i64 %arg)
  %swp2 = tail call <2 x i64> @llvm.bswap.v2i64(<2 x i64> %arg2)
  %swp4 = tail call <4 x i64> @llvm.bswap.v4i64(<4 x i64> undef)
  ret void
}

define void @bswap_i32(i32 %arg, <2 x i32> %arg2, <4 x i32> %arg4) {
; CHECK: Printing analysis 'Cost Model Analysis' for function 'bswap_i32':
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %swp1 = tail call i32
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %swp2 = tail call <2 x i32>
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %swp4 = tail call <4 x i32>
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %swp8 = tail call <8 x i32>
  %swp1 = tail call i32 @llvm.bswap.i32(i32 %arg)
  %swp2 = tail call <2 x i32> @llvm.bswap.v2i32(<2 x i32> %arg2)
  %swp4 = tail call <4 x i32> @llvm.bswap.v4i32(<4 x i32> %arg4)
  %swp8 = tail call <8 x i32> @llvm.bswap.v8i32(<8 x i32> undef)
  ret void
}

define void @bswap_i16(i16 %arg, <2 x i16> %arg2, <4 x i16> %arg4,
                       <8 x i16> %arg8) {
; CHECK: Printing analysis 'Cost Model Analysis' for function 'bswap_i16':
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %swp1 = tail call i16 @llvm.bswap.i16(i16 %arg)
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %swp2 = tail call <2 x i16> @llvm.bswap.v2i16(<2 x i16> %arg2)
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %swp4 = tail call <4 x i16> @llvm.bswap.v4i16(<4 x i16> %arg4)
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %swp8 = tail call <8 x i16> @llvm.bswap.v8i16(<8 x i16> %arg8)
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %swp16 = tail call <16 x i16> @llvm.bswap.v16i16(<16 x i16> undef)
  %swp1 = tail call i16 @llvm.bswap.i16(i16 %arg)
  %swp2 = tail call <2 x i16> @llvm.bswap.v2i16(<2 x i16> %arg2)
  %swp4 = tail call <4 x i16> @llvm.bswap.v4i16(<4 x i16> %arg4)
  %swp8 = tail call <8 x i16> @llvm.bswap.v8i16(<8 x i16> %arg8)
  %swp16 = tail call <16 x i16> @llvm.bswap.v16i16(<16 x i16> undef)
  ret void
}


declare i64 @llvm.bswap.i64(i64)
declare <2 x i64> @llvm.bswap.v2i64(<2 x i64>)
declare <4 x i64> @llvm.bswap.v4i64(<4 x i64>)

declare i32 @llvm.bswap.i32(i32)
declare <2 x i32> @llvm.bswap.v2i32(<2 x i32>)
declare <4 x i32> @llvm.bswap.v4i32(<4 x i32>)
declare <8 x i32> @llvm.bswap.v8i32(<8 x i32>)

declare i16 @llvm.bswap.i16(i16)
declare <2 x i16> @llvm.bswap.v2i16(<2 x i16>)
declare <4 x i16> @llvm.bswap.v4i16(<4 x i16>)
declare <8 x i16> @llvm.bswap.v8i16(<8 x i16>)
declare <16 x i16> @llvm.bswap.v16i16(<16 x i16>)
