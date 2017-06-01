; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple | FileCheck %s

define <16 x i8> @foov16i8(<8 x i16> %a0, <8 x i16> %b0) nounwind readnone ssp {
; CHECK-LABEL: foov16i8:
  %vshrn_low_shift = lshr <8 x i16> %a0, <i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5>
  %vshrn_low = trunc <8 x i16> %vshrn_low_shift to <8 x i8>
  %vshrn_high_shift = lshr <8 x i16> %b0, <i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5>
  %vshrn_high = trunc <8 x i16> %vshrn_high_shift to <8 x i8>
; CHECK: shrn.8b v0, v0, #5
; CHECK-NEXT: shrn2.16b v0, v1, #5
; CHECK-NEXT: ret
  %1 = bitcast <8 x i8> %vshrn_low to <1 x i64>
  %2 = bitcast <8 x i8> %vshrn_high to <1 x i64>
  %shuffle.i = shufflevector <1 x i64> %1, <1 x i64> %2, <2 x i32> <i32 0, i32 1>
  %3 = bitcast <2 x i64> %shuffle.i to <16 x i8>
  ret <16 x i8> %3
}

define <8 x i16> @foov8i16(<4 x i32> %a0, <4 x i32> %b0) nounwind readnone ssp {
; CHECK-LABEL: foov8i16:
  %vshrn_low_shift = lshr <4 x i32> %a0, <i32 5, i32 5, i32 5, i32 5>
  %vshrn_low = trunc <4 x i32> %vshrn_low_shift to <4 x i16>
  %vshrn_high_shift = lshr <4 x i32> %b0, <i32 5, i32 5, i32 5, i32 5>
  %vshrn_high = trunc <4 x i32> %vshrn_high_shift to <4 x i16>
; CHECK: shrn.4h v0, v0, #5
; CHECK-NEXT: shrn2.8h v0, v1, #5
; CHECK-NEXT: ret
  %1 = bitcast <4 x i16> %vshrn_low to <1 x i64>
  %2 = bitcast <4 x i16> %vshrn_high to <1 x i64>
  %shuffle.i = shufflevector <1 x i64> %1, <1 x i64> %2, <2 x i32> <i32 0, i32 1>
  %3 = bitcast <2 x i64> %shuffle.i to <8 x i16>
  ret <8 x i16> %3
}

define <4 x i32> @foov4i32(<2 x i64> %a0, <2 x i64> %b0) nounwind readnone ssp {
; CHECK-LABEL: foov4i32:
  %vshrn_low_shift = lshr <2 x i64> %a0, <i64 5, i64 5>
  %vshrn_low = trunc <2 x i64> %vshrn_low_shift to <2 x i32>
  %vshrn_high_shift = lshr <2 x i64> %b0, <i64 5, i64 5>
  %vshrn_high = trunc <2 x i64> %vshrn_high_shift to <2 x i32>
; CHECK: shrn.2s v0, v0, #5
; CHECK-NEXT: shrn2.4s v0, v1, #5
; CHECK-NEXT: ret
  %1 = bitcast <2 x i32> %vshrn_low to <1 x i64>
  %2 = bitcast <2 x i32> %vshrn_high to <1 x i64>
  %shuffle.i = shufflevector <1 x i64> %1, <1 x i64> %2, <2 x i32> <i32 0, i32 1>
  %3 = bitcast <2 x i64> %shuffle.i to <4 x i32>
  ret <4 x i32> %3
}

define <8 x i16> @bar(<4 x i32> %a0, <4 x i32> %a1, <4 x i32> %b0, <4 x i32> %b1) nounwind readnone ssp {
; CHECK-LABEL: bar:
  %vaddhn2.i = tail call <4 x i16> @llvm.aarch64.neon.addhn.v4i16(<4 x i32> %a0, <4 x i32> %a1) nounwind
  %vaddhn2.i10 = tail call <4 x i16> @llvm.aarch64.neon.addhn.v4i16(<4 x i32> %b0, <4 x i32> %b1) nounwind
; CHECK: addhn.4h	v0, v0, v1
; CHECK-NEXT: addhn2.8h	v0, v2, v3
; CHECK-NEXT: ret
  %1 = bitcast <4 x i16> %vaddhn2.i to <1 x i64>
  %2 = bitcast <4 x i16> %vaddhn2.i10 to <1 x i64>
  %shuffle.i = shufflevector <1 x i64> %1, <1 x i64> %2, <2 x i32> <i32 0, i32 1>
  %3 = bitcast <2 x i64> %shuffle.i to <8 x i16>
  ret <8 x i16> %3
}

define <8 x i16> @baz(<4 x i32> %a0, <4 x i32> %a1, <4 x i32> %b0, <4 x i32> %b1) nounwind readnone ssp {
; CHECK-LABEL: baz:
  %vaddhn2.i = tail call <4 x i16> @llvm.aarch64.neon.addhn.v4i16(<4 x i32> %a0, <4 x i32> %a1) nounwind
  %vshrn_high_shift = ashr <4 x i32> %b0, <i32 5, i32 5, i32 5, i32 5>
  %vshrn_high = trunc <4 x i32> %vshrn_high_shift to <4 x i16>
; CHECK: addhn.4h	v0, v0, v1
; CHECK-NEXT: shrn2.8h	v0, v2, #5
; CHECK-NEXT: ret
  %1 = bitcast <4 x i16> %vaddhn2.i to <1 x i64>
  %2 = bitcast <4 x i16> %vshrn_high to <1 x i64>
  %shuffle.i = shufflevector <1 x i64> %1, <1 x i64> %2, <2 x i32> <i32 0, i32 1>
  %3 = bitcast <2 x i64> %shuffle.i to <8 x i16>
  ret <8 x i16> %3
}

define <8 x i16> @raddhn(<4 x i32> %a0, <4 x i32> %a1, <4 x i32> %b0, <4 x i32> %b1) nounwind readnone ssp {
; CHECK-LABEL: raddhn:
entry:
; CHECK: 	raddhn.4h	v0, v0, v1
; CHECK-NEXT: 	raddhn2.8h	v0, v2, v3
; CHECK-NEXT: 	ret
  %vraddhn2.i = tail call <4 x i16> @llvm.aarch64.neon.raddhn.v4i16(<4 x i32> %a0, <4 x i32> %a1) nounwind
  %vraddhn2.i10 = tail call <4 x i16> @llvm.aarch64.neon.raddhn.v4i16(<4 x i32> %b0, <4 x i32> %b1) nounwind
  %0 = bitcast <4 x i16> %vraddhn2.i to <1 x i64>
  %1 = bitcast <4 x i16> %vraddhn2.i10 to <1 x i64>
  %shuffle.i = shufflevector <1 x i64> %0, <1 x i64> %1, <2 x i32> <i32 0, i32 1>
  %2 = bitcast <2 x i64> %shuffle.i to <8 x i16>
  ret <8 x i16> %2
}

define <8 x i16> @vrshrn(<8 x i16> %a0, <8 x i16> %a1, <8 x i16> %b0, <8 x i16> %b1) nounwind readnone ssp {
; CHECK-LABEL: vrshrn:
; CHECK: rshrn.8b	v0, v0, #5
; CHECK-NEXT: rshrn2.16b	v0, v2, #6
; CHECK-NEXT: ret
  %vrshrn_n1 = tail call <8 x i8> @llvm.aarch64.neon.rshrn.v8i8(<8 x i16> %a0, i32 5)
  %vrshrn_n4 = tail call <8 x i8> @llvm.aarch64.neon.rshrn.v8i8(<8 x i16> %b0, i32 6)
  %1 = bitcast <8 x i8> %vrshrn_n1 to <1 x i64>
  %2 = bitcast <8 x i8> %vrshrn_n4 to <1 x i64>
  %shuffle.i = shufflevector <1 x i64> %1, <1 x i64> %2, <2 x i32> <i32 0, i32 1>
  %3 = bitcast <2 x i64> %shuffle.i to <8 x i16>
  ret <8 x i16> %3
}

define <8 x i16> @vrsubhn(<8 x i16> %a0, <8 x i16> %a1, <8 x i16> %b0, <8 x i16> %b1) nounwind readnone ssp {
; CHECK-LABEL: vrsubhn:
; CHECK: rsubhn.8b	v0, v0, v1
; CHECK: rsubhn2.16b	v0, v2, v3
; CHECK-NEXT: 	ret
  %vrsubhn2.i = tail call <8 x i8> @llvm.aarch64.neon.rsubhn.v8i8(<8 x i16> %a0, <8 x i16> %a1) nounwind
  %vrsubhn2.i10 = tail call <8 x i8> @llvm.aarch64.neon.rsubhn.v8i8(<8 x i16> %b0, <8 x i16> %b1) nounwind
  %1 = bitcast <8 x i8> %vrsubhn2.i to <1 x i64>
  %2 = bitcast <8 x i8> %vrsubhn2.i10 to <1 x i64>
  %shuffle.i = shufflevector <1 x i64> %1, <1 x i64> %2, <2 x i32> <i32 0, i32 1>
  %3 = bitcast <2 x i64> %shuffle.i to <8 x i16>
  ret <8 x i16> %3
}

define <8 x i16> @noOpt1(<2 x i32> %a0, <2 x i32> %a1, <4 x i32> %b0, <4 x i32> %b1) nounwind readnone ssp {
; CHECK-LABEL: noOpt1:
  %vqsub2.i = tail call <2 x i32> @llvm.aarch64.neon.sqsub.v2i32(<2 x i32> %a0, <2 x i32> %a1) nounwind
  %vaddhn2.i = tail call <4 x i16> @llvm.aarch64.neon.addhn.v4i16(<4 x i32> %b0, <4 x i32> %b1) nounwind
; CHECK:	sqsub.2s	v0, v0, v1
; CHECK-NEXT:	addhn2.8h	v0, v2, v3
  %1 = bitcast <2 x i32> %vqsub2.i to <1 x i64>
  %2 = bitcast <4 x i16> %vaddhn2.i to <1 x i64>
  %shuffle.i = shufflevector <1 x i64> %1, <1 x i64> %2, <2 x i32> <i32 0, i32 1>
  %3 = bitcast <2 x i64> %shuffle.i to <8 x i16>
  ret <8 x i16> %3
}

declare <2 x i32> @llvm.aarch64.neon.sqsub.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

declare <8 x i8> @llvm.aarch64.neon.shrn.v8i8(<8 x i16>, i32) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.shrn.v4i16(<4 x i32>, i32) nounwind readnone
declare <2 x i32> @llvm.aarch64.neon.shrn.v2i32(<2 x i64>, i32) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.addhn.v4i16(<4 x i32>, <4 x i32>) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.raddhn.v4i16(<4 x i32>, <4 x i32>) nounwind readnone
declare <8 x i8> @llvm.aarch64.neon.rshrn.v8i8(<8 x i16>, i32) nounwind readnone
declare <8 x i8> @llvm.aarch64.neon.rsubhn.v8i8(<8 x i16>, <8 x i16>) nounwind readnone

