; RUN: llc -mtriple=arm-eabi -mattr=+neon %s -o - | FileCheck %s

define <8 x i8> @vqadds8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vqadds8:
;CHECK: vqadd.s8
	%tmp1 = load <8 x i8>, ptr %A
	%tmp2 = load <8 x i8>, ptr %B
	%tmp3 = call <8 x i8> @llvm.sadd.sat.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <4 x i16> @vqadds16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vqadds16:
;CHECK: vqadd.s16
	%tmp1 = load <4 x i16>, ptr %A
	%tmp2 = load <4 x i16>, ptr %B
	%tmp3 = call <4 x i16> @llvm.sadd.sat.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @vqadds32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vqadds32:
;CHECK: vqadd.s32
	%tmp1 = load <2 x i32>, ptr %A
	%tmp2 = load <2 x i32>, ptr %B
	%tmp3 = call <2 x i32> @llvm.sadd.sat.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <1 x i64> @vqadds64(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vqadds64:
;CHECK: vqadd.s64
	%tmp1 = load <1 x i64>, ptr %A
	%tmp2 = load <1 x i64>, ptr %B
	%tmp3 = call <1 x i64> @llvm.sadd.sat.v1i64(<1 x i64> %tmp1, <1 x i64> %tmp2)
	ret <1 x i64> %tmp3
}

define <8 x i8> @vqaddu8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vqaddu8:
;CHECK: vqadd.u8
	%tmp1 = load <8 x i8>, ptr %A
	%tmp2 = load <8 x i8>, ptr %B
	%tmp3 = call <8 x i8> @llvm.uadd.sat.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <4 x i16> @vqaddu16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vqaddu16:
;CHECK: vqadd.u16
	%tmp1 = load <4 x i16>, ptr %A
	%tmp2 = load <4 x i16>, ptr %B
	%tmp3 = call <4 x i16> @llvm.uadd.sat.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @vqaddu32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vqaddu32:
;CHECK: vqadd.u32
	%tmp1 = load <2 x i32>, ptr %A
	%tmp2 = load <2 x i32>, ptr %B
	%tmp3 = call <2 x i32> @llvm.uadd.sat.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <1 x i64> @vqaddu64(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vqaddu64:
;CHECK: vqadd.u64
	%tmp1 = load <1 x i64>, ptr %A
	%tmp2 = load <1 x i64>, ptr %B
	%tmp3 = call <1 x i64> @llvm.uadd.sat.v1i64(<1 x i64> %tmp1, <1 x i64> %tmp2)
	ret <1 x i64> %tmp3
}

define <16 x i8> @vqaddQs8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vqaddQs8:
;CHECK: vqadd.s8
	%tmp1 = load <16 x i8>, ptr %A
	%tmp2 = load <16 x i8>, ptr %B
	%tmp3 = call <16 x i8> @llvm.sadd.sat.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

define <8 x i16> @vqaddQs16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vqaddQs16:
;CHECK: vqadd.s16
	%tmp1 = load <8 x i16>, ptr %A
	%tmp2 = load <8 x i16>, ptr %B
	%tmp3 = call <8 x i16> @llvm.sadd.sat.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @vqaddQs32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vqaddQs32:
;CHECK: vqadd.s32
	%tmp1 = load <4 x i32>, ptr %A
	%tmp2 = load <4 x i32>, ptr %B
	%tmp3 = call <4 x i32> @llvm.sadd.sat.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

define <2 x i64> @vqaddQs64(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vqaddQs64:
;CHECK: vqadd.s64
	%tmp1 = load <2 x i64>, ptr %A
	%tmp2 = load <2 x i64>, ptr %B
	%tmp3 = call <2 x i64> @llvm.sadd.sat.v2i64(<2 x i64> %tmp1, <2 x i64> %tmp2)
	ret <2 x i64> %tmp3
}

define <16 x i8> @vqaddQu8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vqaddQu8:
;CHECK: vqadd.u8
	%tmp1 = load <16 x i8>, ptr %A
	%tmp2 = load <16 x i8>, ptr %B
	%tmp3 = call <16 x i8> @llvm.uadd.sat.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

define <8 x i16> @vqaddQu16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vqaddQu16:
;CHECK: vqadd.u16
	%tmp1 = load <8 x i16>, ptr %A
	%tmp2 = load <8 x i16>, ptr %B
	%tmp3 = call <8 x i16> @llvm.uadd.sat.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @vqaddQu32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vqaddQu32:
;CHECK: vqadd.u32
	%tmp1 = load <4 x i32>, ptr %A
	%tmp2 = load <4 x i32>, ptr %B
	%tmp3 = call <4 x i32> @llvm.uadd.sat.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

define <2 x i64> @vqaddQu64(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vqaddQu64:
;CHECK: vqadd.u64
	%tmp1 = load <2 x i64>, ptr %A
	%tmp2 = load <2 x i64>, ptr %B
	%tmp3 = call <2 x i64> @llvm.uadd.sat.v2i64(<2 x i64> %tmp1, <2 x i64> %tmp2)
	ret <2 x i64> %tmp3
}

declare <8 x i8>  @llvm.sadd.sat.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.sadd.sat.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.sadd.sat.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <1 x i64> @llvm.sadd.sat.v1i64(<1 x i64>, <1 x i64>) nounwind readnone

declare <8 x i8>  @llvm.uadd.sat.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.uadd.sat.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.uadd.sat.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <1 x i64> @llvm.uadd.sat.v1i64(<1 x i64>, <1 x i64>) nounwind readnone

declare <16 x i8> @llvm.sadd.sat.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.sadd.sat.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.sadd.sat.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i64> @llvm.sadd.sat.v2i64(<2 x i64>, <2 x i64>) nounwind readnone

declare <16 x i8> @llvm.uadd.sat.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.uadd.sat.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.uadd.sat.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i64> @llvm.uadd.sat.v2i64(<2 x i64>, <2 x i64>) nounwind readnone
