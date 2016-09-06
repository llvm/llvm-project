; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple | FileCheck %s

define <8 x i8> @sqsub8b(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: sqsub8b:
;CHECK: sqsub.8b
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = load <8 x i8>, <8 x i8>* %B
	%tmp3 = call <8 x i8> @llvm.aarch64.neon.sqsub.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <4 x i16> @sqsub4h(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: sqsub4h:
;CHECK: sqsub.4h
	%tmp1 = load <4 x i16>, <4 x i16>* %A
	%tmp2 = load <4 x i16>, <4 x i16>* %B
	%tmp3 = call <4 x i16> @llvm.aarch64.neon.sqsub.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @sqsub2s(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: sqsub2s:
;CHECK: sqsub.2s
	%tmp1 = load <2 x i32>, <2 x i32>* %A
	%tmp2 = load <2 x i32>, <2 x i32>* %B
	%tmp3 = call <2 x i32> @llvm.aarch64.neon.sqsub.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <8 x i8> @uqsub8b(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: uqsub8b:
;CHECK: uqsub.8b
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = load <8 x i8>, <8 x i8>* %B
	%tmp3 = call <8 x i8> @llvm.aarch64.neon.uqsub.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <4 x i16> @uqsub4h(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: uqsub4h:
;CHECK: uqsub.4h
	%tmp1 = load <4 x i16>, <4 x i16>* %A
	%tmp2 = load <4 x i16>, <4 x i16>* %B
	%tmp3 = call <4 x i16> @llvm.aarch64.neon.uqsub.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @uqsub2s(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: uqsub2s:
;CHECK: uqsub.2s
	%tmp1 = load <2 x i32>, <2 x i32>* %A
	%tmp2 = load <2 x i32>, <2 x i32>* %B
	%tmp3 = call <2 x i32> @llvm.aarch64.neon.uqsub.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <16 x i8> @sqsub16b(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: sqsub16b:
;CHECK: sqsub.16b
	%tmp1 = load <16 x i8>, <16 x i8>* %A
	%tmp2 = load <16 x i8>, <16 x i8>* %B
	%tmp3 = call <16 x i8> @llvm.aarch64.neon.sqsub.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

define <8 x i16> @sqsub8h(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: sqsub8h:
;CHECK: sqsub.8h
	%tmp1 = load <8 x i16>, <8 x i16>* %A
	%tmp2 = load <8 x i16>, <8 x i16>* %B
	%tmp3 = call <8 x i16> @llvm.aarch64.neon.sqsub.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @sqsub4s(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: sqsub4s:
;CHECK: sqsub.4s
	%tmp1 = load <4 x i32>, <4 x i32>* %A
	%tmp2 = load <4 x i32>, <4 x i32>* %B
	%tmp3 = call <4 x i32> @llvm.aarch64.neon.sqsub.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

define <2 x i64> @sqsub2d(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK-LABEL: sqsub2d:
;CHECK: sqsub.2d
	%tmp1 = load <2 x i64>, <2 x i64>* %A
	%tmp2 = load <2 x i64>, <2 x i64>* %B
	%tmp3 = call <2 x i64> @llvm.aarch64.neon.sqsub.v2i64(<2 x i64> %tmp1, <2 x i64> %tmp2)
	ret <2 x i64> %tmp3
}

define <16 x i8> @uqsub16b(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: uqsub16b:
;CHECK: uqsub.16b
	%tmp1 = load <16 x i8>, <16 x i8>* %A
	%tmp2 = load <16 x i8>, <16 x i8>* %B
	%tmp3 = call <16 x i8> @llvm.aarch64.neon.uqsub.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

define <8 x i16> @uqsub8h(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: uqsub8h:
;CHECK: uqsub.8h
	%tmp1 = load <8 x i16>, <8 x i16>* %A
	%tmp2 = load <8 x i16>, <8 x i16>* %B
	%tmp3 = call <8 x i16> @llvm.aarch64.neon.uqsub.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @uqsub4s(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: uqsub4s:
;CHECK: uqsub.4s
	%tmp1 = load <4 x i32>, <4 x i32>* %A
	%tmp2 = load <4 x i32>, <4 x i32>* %B
	%tmp3 = call <4 x i32> @llvm.aarch64.neon.uqsub.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

define <2 x i64> @uqsub2d(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK-LABEL: uqsub2d:
;CHECK: uqsub.2d
	%tmp1 = load <2 x i64>, <2 x i64>* %A
	%tmp2 = load <2 x i64>, <2 x i64>* %B
	%tmp3 = call <2 x i64> @llvm.aarch64.neon.uqsub.v2i64(<2 x i64> %tmp1, <2 x i64> %tmp2)
	ret <2 x i64> %tmp3
}

declare <8 x i8>  @llvm.aarch64.neon.sqsub.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.sqsub.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.aarch64.neon.sqsub.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <1 x i64> @llvm.aarch64.neon.sqsub.v1i64(<1 x i64>, <1 x i64>) nounwind readnone

declare <8 x i8>  @llvm.aarch64.neon.uqsub.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.uqsub.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.aarch64.neon.uqsub.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <1 x i64> @llvm.aarch64.neon.uqsub.v1i64(<1 x i64>, <1 x i64>) nounwind readnone

declare <16 x i8> @llvm.aarch64.neon.sqsub.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.aarch64.neon.sqsub.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.aarch64.neon.sqsub.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i64> @llvm.aarch64.neon.sqsub.v2i64(<2 x i64>, <2 x i64>) nounwind readnone

declare <16 x i8> @llvm.aarch64.neon.uqsub.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.aarch64.neon.uqsub.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.aarch64.neon.uqsub.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i64> @llvm.aarch64.neon.uqsub.v2i64(<2 x i64>, <2 x i64>) nounwind readnone
