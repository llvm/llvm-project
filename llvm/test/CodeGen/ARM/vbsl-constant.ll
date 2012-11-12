; RUN: llc < %s -mtriple=arm-apple-ios -mattr=+neon | FileCheck %s

define <8 x i8> @v_bsli8(<8 x i8>* %A, <8 x i8>* %B, <8 x i8>* %C) nounwind {
;CHECK: v_bsli8:
;CHECK: vldr
;CHECK: vldr
;CHECK: vbsl
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = load <8 x i8>* %C
	%tmp4 = and <8 x i8> %tmp1, <i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3>
	%tmp6 = and <8 x i8> %tmp3, <i8 -4, i8 -4, i8 -4, i8 -4, i8 -4, i8 -4, i8 -4, i8 -4>
	%tmp7 = or <8 x i8> %tmp4, %tmp6
	ret <8 x i8> %tmp7
}

define <4 x i16> @v_bsli16(<4 x i16>* %A, <4 x i16>* %B, <4 x i16>* %C) nounwind {
;CHECK: v_bsli16:
;CHECK: vldr
;CHECK: vldr
;CHECK: vbsl
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = load <4 x i16>* %C
	%tmp4 = and <4 x i16> %tmp1, <i16 3, i16 3, i16 3, i16 3>
	%tmp6 = and <4 x i16> %tmp3, <i16 -4, i16 -4, i16 -4, i16 -4>
	%tmp7 = or <4 x i16> %tmp4, %tmp6
	ret <4 x i16> %tmp7
}

define <2 x i32> @v_bsli32(<2 x i32>* %A, <2 x i32>* %B, <2 x i32>* %C) nounwind {
;CHECK: v_bsli32:
;CHECK: vldr
;CHECK: vldr
;CHECK: vbsl
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = load <2 x i32>* %C
	%tmp4 = and <2 x i32> %tmp1, <i32 3, i32 3>
	%tmp6 = and <2 x i32> %tmp3, <i32 -4, i32 -4>
	%tmp7 = or <2 x i32> %tmp4, %tmp6
	ret <2 x i32> %tmp7
}

define <1 x i64> @v_bsli64(<1 x i64>* %A, <1 x i64>* %B, <1 x i64>* %C) nounwind {
;CHECK: v_bsli64:
;CHECK: vldr
;CHECK: vldr
;CHECK: vldr
;CHECK: vbsl
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
	%tmp3 = load <1 x i64>* %C
	%tmp4 = and <1 x i64> %tmp1, <i64 3>
	%tmp6 = and <1 x i64> %tmp3, <i64 -4>
	%tmp7 = or <1 x i64> %tmp4, %tmp6
	ret <1 x i64> %tmp7
}

define <16 x i8> @v_bslQi8(<16 x i8>* %A, <16 x i8>* %B, <16 x i8>* %C) nounwind {
;CHECK: v_bslQi8:
;CHECK: vld1.32
;CHECK: vld1.32
;CHECK: vbsl
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = load <16 x i8>* %C
	%tmp4 = and <16 x i8> %tmp1, <i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3>
	%tmp6 = and <16 x i8> %tmp3, <i8 -4, i8 -4, i8 -4, i8 -4, i8 -4, i8 -4, i8 -4, i8 -4, i8 -4, i8 -4, i8 -4, i8 -4, i8 -4, i8 -4, i8 -4, i8 -4>
	%tmp7 = or <16 x i8> %tmp4, %tmp6
	ret <16 x i8> %tmp7
}

define <8 x i16> @v_bslQi16(<8 x i16>* %A, <8 x i16>* %B, <8 x i16>* %C) nounwind {
;CHECK: v_bslQi16:
;CHECK: vld1.32
;CHECK: vld1.32
;CHECK: vbsl
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = load <8 x i16>* %C
	%tmp4 = and <8 x i16> %tmp1, <i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3>
	%tmp6 = and <8 x i16> %tmp3, <i16 -4, i16 -4, i16 -4, i16 -4, i16 -4, i16 -4, i16 -4, i16 -4>
	%tmp7 = or <8 x i16> %tmp4, %tmp6
	ret <8 x i16> %tmp7
}

define <4 x i32> @v_bslQi32(<4 x i32>* %A, <4 x i32>* %B, <4 x i32>* %C) nounwind {
;CHECK: v_bslQi32:
;CHECK: vld1.32
;CHECK: vld1.32
;CHECK: vbsl
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = load <4 x i32>* %C
	%tmp4 = and <4 x i32> %tmp1, <i32 3, i32 3, i32 3, i32 3>
	%tmp6 = and <4 x i32> %tmp3, <i32 -4, i32 -4, i32 -4, i32 -4>
	%tmp7 = or <4 x i32> %tmp4, %tmp6
	ret <4 x i32> %tmp7
}

define <2 x i64> @v_bslQi64(<2 x i64>* %A, <2 x i64>* %B, <2 x i64>* %C) nounwind {
;CHECK: v_bslQi64:
;CHECK: vld1.32
;CHECK: vld1.32
;CHECK: vld1.64
;CHECK: vbsl
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
	%tmp3 = load <2 x i64>* %C
	%tmp4 = and <2 x i64> %tmp1, <i64 3, i64 3>
	%tmp6 = and <2 x i64> %tmp3, <i64 -4, i64 -4>
	%tmp7 = or <2 x i64> %tmp4, %tmp6
	ret <2 x i64> %tmp7
}
