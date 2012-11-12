; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

define <8 x i8> @vaddi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vaddi8:
;CHECK: vadd.i8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = add <8 x i8> %tmp1, %tmp2
	ret <8 x i8> %tmp3
}

define <4 x i16> @vaddi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vaddi16:
;CHECK: vadd.i16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = add <4 x i16> %tmp1, %tmp2
	ret <4 x i16> %tmp3
}

define <2 x i32> @vaddi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vaddi32:
;CHECK: vadd.i32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = add <2 x i32> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}

define <1 x i64> @vaddi64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
;CHECK: vaddi64:
;CHECK: vadd.i64
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
	%tmp3 = add <1 x i64> %tmp1, %tmp2
	ret <1 x i64> %tmp3
}

define <2 x float> @vaddf32(<2 x float>* %A, <2 x float>* %B) nounwind {
;CHECK: vaddf32:
;CHECK: vadd.f32
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
	%tmp3 = fadd <2 x float> %tmp1, %tmp2
	ret <2 x float> %tmp3
}

define <16 x i8> @vaddQi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK: vaddQi8:
;CHECK: vadd.i8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = add <16 x i8> %tmp1, %tmp2
	ret <16 x i8> %tmp3
}

define <8 x i16> @vaddQi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK: vaddQi16:
;CHECK: vadd.i16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = add <8 x i16> %tmp1, %tmp2
	ret <8 x i16> %tmp3
}

define <4 x i32> @vaddQi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK: vaddQi32:
;CHECK: vadd.i32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = add <4 x i32> %tmp1, %tmp2
	ret <4 x i32> %tmp3
}

define <2 x i64> @vaddQi64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK: vaddQi64:
;CHECK: vadd.i64
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
	%tmp3 = add <2 x i64> %tmp1, %tmp2
	ret <2 x i64> %tmp3
}

define <4 x float> @vaddQf32(<4 x float>* %A, <4 x float>* %B) nounwind {
;CHECK: vaddQf32:
;CHECK: vadd.f32
	%tmp1 = load <4 x float>* %A
	%tmp2 = load <4 x float>* %B
	%tmp3 = fadd <4 x float> %tmp1, %tmp2
	ret <4 x float> %tmp3
}

define <8 x i8> @vaddhni16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK: vaddhni16:
;CHECK: vaddhn.i16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = call <8 x i8> @llvm.arm.neon.vaddhn.v8i8(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i8> %tmp3
}

define <4 x i16> @vaddhni32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK: vaddhni32:
;CHECK: vaddhn.i32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = call <4 x i16> @llvm.arm.neon.vaddhn.v4i16(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @vaddhni64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK: vaddhni64:
;CHECK: vaddhn.i64
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
	%tmp3 = call <2 x i32> @llvm.arm.neon.vaddhn.v2i32(<2 x i64> %tmp1, <2 x i64> %tmp2)
	ret <2 x i32> %tmp3
}

declare <8 x i8>  @llvm.arm.neon.vaddhn.v8i8(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vaddhn.v4i16(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vaddhn.v2i32(<2 x i64>, <2 x i64>) nounwind readnone

define <8 x i8> @vraddhni16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK: vraddhni16:
;CHECK: vraddhn.i16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = call <8 x i8> @llvm.arm.neon.vraddhn.v8i8(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i8> %tmp3
}

define <4 x i16> @vraddhni32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK: vraddhni32:
;CHECK: vraddhn.i32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = call <4 x i16> @llvm.arm.neon.vraddhn.v4i16(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @vraddhni64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK: vraddhni64:
;CHECK: vraddhn.i64
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
	%tmp3 = call <2 x i32> @llvm.arm.neon.vraddhn.v2i32(<2 x i64> %tmp1, <2 x i64> %tmp2)
	ret <2 x i32> %tmp3
}

declare <8 x i8>  @llvm.arm.neon.vraddhn.v8i8(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vraddhn.v4i16(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vraddhn.v2i32(<2 x i64>, <2 x i64>) nounwind readnone

define <8 x i16> @vaddls8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vaddls8:
;CHECK: vaddl.s8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = sext <8 x i8> %tmp1 to <8 x i16>
	%tmp4 = sext <8 x i8> %tmp2 to <8 x i16>
	%tmp5 = add <8 x i16> %tmp3, %tmp4
	ret <8 x i16> %tmp5
}

define <4 x i32> @vaddls16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vaddls16:
;CHECK: vaddl.s16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = sext <4 x i16> %tmp1 to <4 x i32>
	%tmp4 = sext <4 x i16> %tmp2 to <4 x i32>
	%tmp5 = add <4 x i32> %tmp3, %tmp4
	ret <4 x i32> %tmp5
}

define <2 x i64> @vaddls32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vaddls32:
;CHECK: vaddl.s32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = sext <2 x i32> %tmp1 to <2 x i64>
	%tmp4 = sext <2 x i32> %tmp2 to <2 x i64>
	%tmp5 = add <2 x i64> %tmp3, %tmp4
	ret <2 x i64> %tmp5
}

define <8 x i16> @vaddlu8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vaddlu8:
;CHECK: vaddl.u8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = zext <8 x i8> %tmp1 to <8 x i16>
	%tmp4 = zext <8 x i8> %tmp2 to <8 x i16>
	%tmp5 = add <8 x i16> %tmp3, %tmp4
	ret <8 x i16> %tmp5
}

define <4 x i32> @vaddlu16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vaddlu16:
;CHECK: vaddl.u16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = zext <4 x i16> %tmp1 to <4 x i32>
	%tmp4 = zext <4 x i16> %tmp2 to <4 x i32>
	%tmp5 = add <4 x i32> %tmp3, %tmp4
	ret <4 x i32> %tmp5
}

define <2 x i64> @vaddlu32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vaddlu32:
;CHECK: vaddl.u32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = zext <2 x i32> %tmp1 to <2 x i64>
	%tmp4 = zext <2 x i32> %tmp2 to <2 x i64>
	%tmp5 = add <2 x i64> %tmp3, %tmp4
	ret <2 x i64> %tmp5
}

define <8 x i16> @vaddws8(<8 x i16>* %A, <8 x i8>* %B) nounwind {
;CHECK: vaddws8:
;CHECK: vaddw.s8
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = sext <8 x i8> %tmp2 to <8 x i16>
	%tmp4 = add <8 x i16> %tmp1, %tmp3
	ret <8 x i16> %tmp4
}

define <4 x i32> @vaddws16(<4 x i32>* %A, <4 x i16>* %B) nounwind {
;CHECK: vaddws16:
;CHECK: vaddw.s16
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = sext <4 x i16> %tmp2 to <4 x i32>
	%tmp4 = add <4 x i32> %tmp1, %tmp3
	ret <4 x i32> %tmp4
}

define <2 x i64> @vaddws32(<2 x i64>* %A, <2 x i32>* %B) nounwind {
;CHECK: vaddws32:
;CHECK: vaddw.s32
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = sext <2 x i32> %tmp2 to <2 x i64>
	%tmp4 = add <2 x i64> %tmp1, %tmp3
	ret <2 x i64> %tmp4
}

define <8 x i16> @vaddwu8(<8 x i16>* %A, <8 x i8>* %B) nounwind {
;CHECK: vaddwu8:
;CHECK: vaddw.u8
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = zext <8 x i8> %tmp2 to <8 x i16>
	%tmp4 = add <8 x i16> %tmp1, %tmp3
	ret <8 x i16> %tmp4
}

define <4 x i32> @vaddwu16(<4 x i32>* %A, <4 x i16>* %B) nounwind {
;CHECK: vaddwu16:
;CHECK: vaddw.u16
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = zext <4 x i16> %tmp2 to <4 x i32>
	%tmp4 = add <4 x i32> %tmp1, %tmp3
	ret <4 x i32> %tmp4
}

define <2 x i64> @vaddwu32(<2 x i64>* %A, <2 x i32>* %B) nounwind {
;CHECK: vaddwu32:
;CHECK: vaddw.u32
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = zext <2 x i32> %tmp2 to <2 x i64>
	%tmp4 = add <2 x i64> %tmp1, %tmp3
	ret <2 x i64> %tmp4
}
