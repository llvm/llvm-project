; RUN: opt -cost-model -analyze -mtriple=aarch64--linux-gnu < %s | FileCheck %s

; Verify the cost of bswap instructions.

declare i16 @llvm.bswap.i16(i16)
declare i32 @llvm.bswap.i32(i32)
declare i64 @llvm.bswap.i64(i64)

declare <2 x i32> @llvm.bswap.v2i32(<2 x i32>)
declare <4 x i16> @llvm.bswap.v4i16(<4 x i16>)

declare <2 x i64> @llvm.bswap.v2i64(<2 x i64>)
declare <4 x i32> @llvm.bswap.v4i32(<4 x i32>)
declare <8 x i16> @llvm.bswap.v8i16(<8 x i16>)

define i16 @bswap_i16(i16 %a) {
; CHECK: 'Cost Model Analysis' for function 'bswap_i16':
; CHECK: Found an estimated cost of 1 for instruction:   %bswap
  %bswap = tail call i16 @llvm.bswap.i16(i16 %a)
  ret i16 %bswap
}

define i32 @bswap_i32(i32 %a) {
; CHECK: 'Cost Model Analysis' for function 'bswap_i32':
; CHECK: Found an estimated cost of 1 for instruction:   %bswap
  %bswap = tail call i32 @llvm.bswap.i32(i32 %a)
  ret i32 %bswap
}

define i64 @bswap_i64(i64 %a) {
; CHECK: 'Cost Model Analysis' for function 'bswap_i64':
; CHECK: Found an estimated cost of 1 for instruction:   %bswap
  %bswap = tail call i64 @llvm.bswap.i64(i64 %a)
  ret i64 %bswap
}

define <2 x i32> @bswap_v2i32(<2 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'bswap_v2i32':
; CHECK: Found an estimated cost of 8 for instruction:   %bswap
  %bswap = call <2 x i32> @llvm.bswap.v2i32(<2 x i32> %a)
  ret <2 x i32> %bswap
}

define <4 x i16> @bswap_v4i16(<4 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'bswap_v4i16':
; CHECK: Found an estimated cost of 22 for instruction:   %bswap
  %bswap = call <4 x i16> @llvm.bswap.v4i16(<4 x i16> %a)
  ret <4 x i16> %bswap
}

define <2 x i64> @bswap_v2i64(<2 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'bswap_v2i64':
; CHECK: Found an estimated cost of 8 for instruction:   %bswap
  %bswap = call <2 x i64> @llvm.bswap.v2i64(<2 x i64> %a)
  ret <2 x i64> %bswap
}

define <4 x i32> @bswap_v4i32(<4 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'bswap_v4i32':
; CHECK: Found an estimated cost of 22 for instruction:   %bswap
  %bswap = call <4 x i32> @llvm.bswap.v4i32(<4 x i32> %a)
  ret <4 x i32> %bswap
}

define <8 x i16> @bswap_v8i16(<8 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'bswap_v8i16':
; CHECK: Found an estimated cost of 50 for instruction:   %bswap
  %bswap = call <8 x i16> @llvm.bswap.v8i16(<8 x i16> %a)
  ret <8 x i16> %bswap
}
