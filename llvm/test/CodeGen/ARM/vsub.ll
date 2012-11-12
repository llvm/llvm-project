; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

define <8 x i8> @vsubi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vsubi8:
;CHECK: vsub.i8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = sub <8 x i8> %tmp1, %tmp2
	ret <8 x i8> %tmp3
}

define <4 x i16> @vsubi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vsubi16:
;CHECK: vsub.i16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = sub <4 x i16> %tmp1, %tmp2
	ret <4 x i16> %tmp3
}

define <2 x i32> @vsubi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vsubi32:
;CHECK: vsub.i32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = sub <2 x i32> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}

define <1 x i64> @vsubi64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
;CHECK: vsubi64:
;CHECK: vsub.i64
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
	%tmp3 = sub <1 x i64> %tmp1, %tmp2
	ret <1 x i64> %tmp3
}

define <2 x float> @vsubf32(<2 x float>* %A, <2 x float>* %B) nounwind {
;CHECK: vsubf32:
;CHECK: vsub.f32
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
	%tmp3 = fsub <2 x float> %tmp1, %tmp2
	ret <2 x float> %tmp3
}

define <16 x i8> @vsubQi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK: vsubQi8:
;CHECK: vsub.i8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = sub <16 x i8> %tmp1, %tmp2
	ret <16 x i8> %tmp3
}

define <8 x i16> @vsubQi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK: vsubQi16:
;CHECK: vsub.i16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = sub <8 x i16> %tmp1, %tmp2
	ret <8 x i16> %tmp3
}

define <4 x i32> @vsubQi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK: vsubQi32:
;CHECK: vsub.i32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = sub <4 x i32> %tmp1, %tmp2
	ret <4 x i32> %tmp3
}

define <2 x i64> @vsubQi64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK: vsubQi64:
;CHECK: vsub.i64
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
	%tmp3 = sub <2 x i64> %tmp1, %tmp2
	ret <2 x i64> %tmp3
}

define <4 x float> @vsubQf32(<4 x float>* %A, <4 x float>* %B) nounwind {
;CHECK: vsubQf32:
;CHECK: vsub.f32
	%tmp1 = load <4 x float>* %A
	%tmp2 = load <4 x float>* %B
	%tmp3 = fsub <4 x float> %tmp1, %tmp2
	ret <4 x float> %tmp3
}

define <8 x i8> @vsubhni16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK: vsubhni16:
;CHECK: vsubhn.i16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = call <8 x i8> @llvm.arm.neon.vsubhn.v8i8(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i8> %tmp3
}

define <4 x i16> @vsubhni32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK: vsubhni32:
;CHECK: vsubhn.i32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = call <4 x i16> @llvm.arm.neon.vsubhn.v4i16(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @vsubhni64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK: vsubhni64:
;CHECK: vsubhn.i64
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
	%tmp3 = call <2 x i32> @llvm.arm.neon.vsubhn.v2i32(<2 x i64> %tmp1, <2 x i64> %tmp2)
	ret <2 x i32> %tmp3
}

declare <8 x i8>  @llvm.arm.neon.vsubhn.v8i8(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vsubhn.v4i16(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vsubhn.v2i32(<2 x i64>, <2 x i64>) nounwind readnone

define <8 x i8> @vrsubhni16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK: vrsubhni16:
;CHECK: vrsubhn.i16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = call <8 x i8> @llvm.arm.neon.vrsubhn.v8i8(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i8> %tmp3
}

define <4 x i16> @vrsubhni32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK: vrsubhni32:
;CHECK: vrsubhn.i32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = call <4 x i16> @llvm.arm.neon.vrsubhn.v4i16(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @vrsubhni64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK: vrsubhni64:
;CHECK: vrsubhn.i64
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
	%tmp3 = call <2 x i32> @llvm.arm.neon.vrsubhn.v2i32(<2 x i64> %tmp1, <2 x i64> %tmp2)
	ret <2 x i32> %tmp3
}

declare <8 x i8>  @llvm.arm.neon.vrsubhn.v8i8(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vrsubhn.v4i16(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vrsubhn.v2i32(<2 x i64>, <2 x i64>) nounwind readnone

define <8 x i16> @vsubls8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vsubls8:
;CHECK: vsubl.s8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = sext <8 x i8> %tmp1 to <8 x i16>
	%tmp4 = sext <8 x i8> %tmp2 to <8 x i16>
	%tmp5 = sub <8 x i16> %tmp3, %tmp4
	ret <8 x i16> %tmp5
}

define <4 x i32> @vsubls16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vsubls16:
;CHECK: vsubl.s16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = sext <4 x i16> %tmp1 to <4 x i32>
	%tmp4 = sext <4 x i16> %tmp2 to <4 x i32>
	%tmp5 = sub <4 x i32> %tmp3, %tmp4
	ret <4 x i32> %tmp5
}

define <2 x i64> @vsubls32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vsubls32:
;CHECK: vsubl.s32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = sext <2 x i32> %tmp1 to <2 x i64>
	%tmp4 = sext <2 x i32> %tmp2 to <2 x i64>
	%tmp5 = sub <2 x i64> %tmp3, %tmp4
	ret <2 x i64> %tmp5
}

define <8 x i16> @vsublu8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vsublu8:
;CHECK: vsubl.u8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = zext <8 x i8> %tmp1 to <8 x i16>
	%tmp4 = zext <8 x i8> %tmp2 to <8 x i16>
	%tmp5 = sub <8 x i16> %tmp3, %tmp4
	ret <8 x i16> %tmp5
}

define <4 x i32> @vsublu16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vsublu16:
;CHECK: vsubl.u16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = zext <4 x i16> %tmp1 to <4 x i32>
	%tmp4 = zext <4 x i16> %tmp2 to <4 x i32>
	%tmp5 = sub <4 x i32> %tmp3, %tmp4
	ret <4 x i32> %tmp5
}

define <2 x i64> @vsublu32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vsublu32:
;CHECK: vsubl.u32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = zext <2 x i32> %tmp1 to <2 x i64>
	%tmp4 = zext <2 x i32> %tmp2 to <2 x i64>
	%tmp5 = sub <2 x i64> %tmp3, %tmp4
	ret <2 x i64> %tmp5
}

define <8 x i16> @vsubws8(<8 x i16>* %A, <8 x i8>* %B) nounwind {
;CHECK: vsubws8:
;CHECK: vsubw.s8
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = sext <8 x i8> %tmp2 to <8 x i16>
	%tmp4 = sub <8 x i16> %tmp1, %tmp3
	ret <8 x i16> %tmp4
}

define <4 x i32> @vsubws16(<4 x i32>* %A, <4 x i16>* %B) nounwind {
;CHECK: vsubws16:
;CHECK: vsubw.s16
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = sext <4 x i16> %tmp2 to <4 x i32>
	%tmp4 = sub <4 x i32> %tmp1, %tmp3
	ret <4 x i32> %tmp4
}

define <2 x i64> @vsubws32(<2 x i64>* %A, <2 x i32>* %B) nounwind {
;CHECK: vsubws32:
;CHECK: vsubw.s32
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = sext <2 x i32> %tmp2 to <2 x i64>
	%tmp4 = sub <2 x i64> %tmp1, %tmp3
	ret <2 x i64> %tmp4
}

define <8 x i16> @vsubwu8(<8 x i16>* %A, <8 x i8>* %B) nounwind {
;CHECK: vsubwu8:
;CHECK: vsubw.u8
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = zext <8 x i8> %tmp2 to <8 x i16>
	%tmp4 = sub <8 x i16> %tmp1, %tmp3
	ret <8 x i16> %tmp4
}

define <4 x i32> @vsubwu16(<4 x i32>* %A, <4 x i16>* %B) nounwind {
;CHECK: vsubwu16:
;CHECK: vsubw.u16
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = zext <4 x i16> %tmp2 to <4 x i32>
	%tmp4 = sub <4 x i32> %tmp1, %tmp3
	ret <4 x i32> %tmp4
}

define <2 x i64> @vsubwu32(<2 x i64>* %A, <2 x i32>* %B) nounwind {
;CHECK: vsubwu32:
;CHECK: vsubw.u32
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = zext <2 x i32> %tmp2 to <2 x i64>
	%tmp4 = sub <2 x i64> %tmp1, %tmp3
	ret <2 x i64> %tmp4
}
