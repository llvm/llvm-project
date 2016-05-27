; RUN: llc < %s -march=arm64 -aarch64-neon-syntax=apple | FileCheck %s

define i32 @test_rev_w(i32 %a) nounwind {
entry:
; CHECK-LABEL: test_rev_w:
; CHECK: rev w0, w0
  %0 = tail call i32 @llvm.bswap.i32(i32 %a)
  ret i32 %0
}

define i64 @test_rev_x(i64 %a) nounwind {
entry:
; CHECK-LABEL: test_rev_x:
; CHECK: rev x0, x0
  %0 = tail call i64 @llvm.bswap.i64(i64 %a)
  ret i64 %0
}

; Canonicalize (srl (bswap x), 16) to (rotr (bswap x), 16) if the high 16-bits
; of %a are zero. This optimizes rev + lsr 16 to rev16.
define i32 @test_rev_w_srl16(i16 %a) {
entry:
; CHECK-LABEL: test_rev_w_srl16:
; CHECK: and [[REG:w[0-9]+]], w0, #0xffff
; CHECK: rev16 w0, [[REG]]
; CHECK-NOT: lsr
  %0 = zext i16 %a to i32
  %1 = tail call i32 @llvm.bswap.i32(i32 %0)
  %2 = lshr i32 %1, 16
  ret i32 %2
}

; Canonicalize (srl (bswap x), 32) to (rotr (bswap x), 32) if the high 32-bits
; of %a are zero. This optimizes rev + lsr 32 to rev32.
define i64 @test_rev_x_srl32(i32 %a) {
entry:
; CHECK-LABEL: test_rev_x_srl32:
; CHECK: rev32 x0, {{x[0-9]+}}
; CHECK-NOT: lsr
  %0 = zext i32 %a to i64
  %1 = tail call i64 @llvm.bswap.i64(i64 %0)
  %2 = lshr i64 %1, 32
  ret i64 %2
}

declare i32 @llvm.bswap.i32(i32) nounwind readnone
declare i64 @llvm.bswap.i64(i64) nounwind readnone

define i32 @test_rev16_w(i32 %X) nounwind {
entry:
; CHECK-LABEL: test_rev16_w:
; CHECK: rev16 w0, w0
  %tmp1 = lshr i32 %X, 8
  %X15 = bitcast i32 %X to i32
  %tmp4 = shl i32 %X15, 8
  %tmp2 = and i32 %tmp1, 16711680
  %tmp5 = and i32 %tmp4, -16777216
  %tmp9 = and i32 %tmp1, 255
  %tmp13 = and i32 %tmp4, 65280
  %tmp6 = or i32 %tmp5, %tmp2
  %tmp10 = or i32 %tmp6, %tmp13
  %tmp14 = or i32 %tmp10, %tmp9
  ret i32 %tmp14
}

; 64-bit REV16 is *not* a swap then a 16-bit rotation:
;   01234567 ->(bswap) 76543210 ->(rotr) 10765432
;   01234567 ->(rev16) 10325476
define i64 @test_rev16_x(i64 %a) nounwind {
entry:
; CHECK-LABEL: test_rev16_x:
; CHECK-NOT: rev16 x0, x0
  %0 = tail call i64 @llvm.bswap.i64(i64 %a)
  %1 = lshr i64 %0, 16
  %2 = shl i64 %0, 48
  %3 = or i64 %1, %2
  ret i64 %3
}

define i64 @test_rev32_x(i64 %a) nounwind {
entry:
; CHECK-LABEL: test_rev32_x:
; CHECK: rev32 x0, x0
  %0 = tail call i64 @llvm.bswap.i64(i64 %a)
  %1 = lshr i64 %0, 32
  %2 = shl i64 %0, 32
  %3 = or i64 %1, %2
  ret i64 %3
}

define <8 x i8> @test_vrev64D8(<8 x i8>* %A) nounwind {
;CHECK-LABEL: test_vrev64D8:
;CHECK: rev64.8b
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = shufflevector <8 x i8> %tmp1, <8 x i8> undef, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
	ret <8 x i8> %tmp2
}

define <4 x i16> @test_vrev64D16(<4 x i16>* %A) nounwind {
;CHECK-LABEL: test_vrev64D16:
;CHECK: rev64.4h
	%tmp1 = load <4 x i16>, <4 x i16>* %A
	%tmp2 = shufflevector <4 x i16> %tmp1, <4 x i16> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
	ret <4 x i16> %tmp2
}

define <2 x i32> @test_vrev64D32(<2 x i32>* %A) nounwind {
;CHECK-LABEL: test_vrev64D32:
;CHECK: rev64.2s
	%tmp1 = load <2 x i32>, <2 x i32>* %A
	%tmp2 = shufflevector <2 x i32> %tmp1, <2 x i32> undef, <2 x i32> <i32 1, i32 0>
	ret <2 x i32> %tmp2
}

define <2 x float> @test_vrev64Df(<2 x float>* %A) nounwind {
;CHECK-LABEL: test_vrev64Df:
;CHECK: rev64.2s
	%tmp1 = load <2 x float>, <2 x float>* %A
	%tmp2 = shufflevector <2 x float> %tmp1, <2 x float> undef, <2 x i32> <i32 1, i32 0>
	ret <2 x float> %tmp2
}

define <16 x i8> @test_vrev64Q8(<16 x i8>* %A) nounwind {
;CHECK-LABEL: test_vrev64Q8:
;CHECK: rev64.16b
	%tmp1 = load <16 x i8>, <16 x i8>* %A
	%tmp2 = shufflevector <16 x i8> %tmp1, <16 x i8> undef, <16 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0, i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8>
	ret <16 x i8> %tmp2
}

define <8 x i16> @test_vrev64Q16(<8 x i16>* %A) nounwind {
;CHECK-LABEL: test_vrev64Q16:
;CHECK: rev64.8h
	%tmp1 = load <8 x i16>, <8 x i16>* %A
	%tmp2 = shufflevector <8 x i16> %tmp1, <8 x i16> undef, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4>
	ret <8 x i16> %tmp2
}

define <4 x i32> @test_vrev64Q32(<4 x i32>* %A) nounwind {
;CHECK-LABEL: test_vrev64Q32:
;CHECK: rev64.4s
	%tmp1 = load <4 x i32>, <4 x i32>* %A
	%tmp2 = shufflevector <4 x i32> %tmp1, <4 x i32> undef, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
	ret <4 x i32> %tmp2
}

define <4 x float> @test_vrev64Qf(<4 x float>* %A) nounwind {
;CHECK-LABEL: test_vrev64Qf:
;CHECK: rev64.4s
	%tmp1 = load <4 x float>, <4 x float>* %A
	%tmp2 = shufflevector <4 x float> %tmp1, <4 x float> undef, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
	ret <4 x float> %tmp2
}

define <8 x i8> @test_vrev32D8(<8 x i8>* %A) nounwind {
;CHECK-LABEL: test_vrev32D8:
;CHECK: rev32.8b
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = shufflevector <8 x i8> %tmp1, <8 x i8> undef, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4>
	ret <8 x i8> %tmp2
}

define <4 x i16> @test_vrev32D16(<4 x i16>* %A) nounwind {
;CHECK-LABEL: test_vrev32D16:
;CHECK: rev32.4h
	%tmp1 = load <4 x i16>, <4 x i16>* %A
	%tmp2 = shufflevector <4 x i16> %tmp1, <4 x i16> undef, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
	ret <4 x i16> %tmp2
}

define <16 x i8> @test_vrev32Q8(<16 x i8>* %A) nounwind {
;CHECK-LABEL: test_vrev32Q8:
;CHECK: rev32.16b
	%tmp1 = load <16 x i8>, <16 x i8>* %A
	%tmp2 = shufflevector <16 x i8> %tmp1, <16 x i8> undef, <16 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4, i32 11, i32 10, i32 9, i32 8, i32 15, i32 14, i32 13, i32 12>
	ret <16 x i8> %tmp2
}

define <8 x i16> @test_vrev32Q16(<8 x i16>* %A) nounwind {
;CHECK-LABEL: test_vrev32Q16:
;CHECK: rev32.8h
	%tmp1 = load <8 x i16>, <8 x i16>* %A
	%tmp2 = shufflevector <8 x i16> %tmp1, <8 x i16> undef, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6>
	ret <8 x i16> %tmp2
}

define <8 x i8> @test_vrev16D8(<8 x i8>* %A) nounwind {
;CHECK-LABEL: test_vrev16D8:
;CHECK: rev16.8b
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = shufflevector <8 x i8> %tmp1, <8 x i8> undef, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6>
	ret <8 x i8> %tmp2
}

define <16 x i8> @test_vrev16Q8(<16 x i8>* %A) nounwind {
;CHECK-LABEL: test_vrev16Q8:
;CHECK: rev16.16b
	%tmp1 = load <16 x i8>, <16 x i8>* %A
	%tmp2 = shufflevector <16 x i8> %tmp1, <16 x i8> undef, <16 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6, i32 9, i32 8, i32 11, i32 10, i32 13, i32 12, i32 15, i32 14>
	ret <16 x i8> %tmp2
}

; Undef shuffle indices should not prevent matching to VREV:

define <8 x i8> @test_vrev64D8_undef(<8 x i8>* %A) nounwind {
;CHECK-LABEL: test_vrev64D8_undef:
;CHECK: rev64.8b
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = shufflevector <8 x i8> %tmp1, <8 x i8> undef, <8 x i32> <i32 7, i32 undef, i32 undef, i32 4, i32 3, i32 2, i32 1, i32 0>
	ret <8 x i8> %tmp2
}

define <8 x i16> @test_vrev32Q16_undef(<8 x i16>* %A) nounwind {
;CHECK-LABEL: test_vrev32Q16_undef:
;CHECK: rev32.8h
	%tmp1 = load <8 x i16>, <8 x i16>* %A
	%tmp2 = shufflevector <8 x i16> %tmp1, <8 x i16> undef, <8 x i32> <i32 undef, i32 0, i32 undef, i32 2, i32 5, i32 4, i32 7, i32 undef>
	ret <8 x i16> %tmp2
}

; vrev <4 x i16> should use REV32 and not REV64
define void @test_vrev64(<4 x i16>* nocapture %source, <2 x i16>* nocapture %dst) nounwind ssp {
; CHECK-LABEL: test_vrev64:
; CHECK: ldr [[DEST:q[0-9]+]],
; CHECK: st1.h
; CHECK: st1.h
entry:
  %0 = bitcast <4 x i16>* %source to <8 x i16>*
  %tmp2 = load <8 x i16>, <8 x i16>* %0, align 4
  %tmp3 = extractelement <8 x i16> %tmp2, i32 6
  %tmp5 = insertelement <2 x i16> undef, i16 %tmp3, i32 0
  %tmp9 = extractelement <8 x i16> %tmp2, i32 5
  %tmp11 = insertelement <2 x i16> %tmp5, i16 %tmp9, i32 1
  store <2 x i16> %tmp11, <2 x i16>* %dst, align 4
  ret void
}

; Test vrev of float4
define void @float_vrev64(float* nocapture %source, <4 x float>* nocapture %dest) nounwind noinline ssp {
; CHECK: float_vrev64
; CHECK: ldr [[DEST:q[0-9]+]],
; CHECK: rev64.4s
entry:
  %0 = bitcast float* %source to <4 x float>*
  %tmp2 = load <4 x float>, <4 x float>* %0, align 4
  %tmp5 = shufflevector <4 x float> <float 0.000000e+00, float undef, float undef, float undef>, <4 x float> %tmp2, <4 x i32> <i32 0, i32 7, i32 0, i32 0>
  %arrayidx8 = getelementptr inbounds <4 x float>, <4 x float>* %dest, i32 11
  store <4 x float> %tmp5, <4 x float>* %arrayidx8, align 4
  ret void
}


define <4 x i32> @test_vrev32_bswap(<4 x i32> %source) nounwind {
; CHECK-LABEL: test_vrev32_bswap:
; CHECK: rev32.16b
; CHECK-NOT: rev
; CHECK: ret
  %bswap = call <4 x i32> @llvm.bswap.v4i32(<4 x i32> %source)
  ret <4 x i32> %bswap
}

declare <4 x i32> @llvm.bswap.v4i32(<4 x i32>) nounwind readnone
