; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple | FileCheck %s

define <8 x i8> @shadd8b(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: shadd8b:
;CHECK: shadd.8b
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = load <8 x i8>, <8 x i8>* %B
	%tmp3 = call <8 x i8> @llvm.aarch64.neon.shadd.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <16 x i8> @shadd16b(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: shadd16b:
;CHECK: shadd.16b
	%tmp1 = load <16 x i8>, <16 x i8>* %A
	%tmp2 = load <16 x i8>, <16 x i8>* %B
	%tmp3 = call <16 x i8> @llvm.aarch64.neon.shadd.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

define <4 x i16> @shadd4h(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: shadd4h:
;CHECK: shadd.4h
	%tmp1 = load <4 x i16>, <4 x i16>* %A
	%tmp2 = load <4 x i16>, <4 x i16>* %B
	%tmp3 = call <4 x i16> @llvm.aarch64.neon.shadd.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <8 x i16> @shadd8h(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: shadd8h:
;CHECK: shadd.8h
	%tmp1 = load <8 x i16>, <8 x i16>* %A
	%tmp2 = load <8 x i16>, <8 x i16>* %B
	%tmp3 = call <8 x i16> @llvm.aarch64.neon.shadd.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <2 x i32> @shadd2s(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: shadd2s:
;CHECK: shadd.2s
	%tmp1 = load <2 x i32>, <2 x i32>* %A
	%tmp2 = load <2 x i32>, <2 x i32>* %B
	%tmp3 = call <2 x i32> @llvm.aarch64.neon.shadd.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <4 x i32> @shadd4s(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: shadd4s:
;CHECK: shadd.4s
	%tmp1 = load <4 x i32>, <4 x i32>* %A
	%tmp2 = load <4 x i32>, <4 x i32>* %B
	%tmp3 = call <4 x i32> @llvm.aarch64.neon.shadd.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

define <8 x i8> @uhadd8b(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: uhadd8b:
;CHECK: uhadd.8b
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = load <8 x i8>, <8 x i8>* %B
	%tmp3 = call <8 x i8> @llvm.aarch64.neon.uhadd.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <16 x i8> @uhadd16b(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: uhadd16b:
;CHECK: uhadd.16b
	%tmp1 = load <16 x i8>, <16 x i8>* %A
	%tmp2 = load <16 x i8>, <16 x i8>* %B
	%tmp3 = call <16 x i8> @llvm.aarch64.neon.uhadd.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

define <4 x i16> @uhadd4h(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: uhadd4h:
;CHECK: uhadd.4h
	%tmp1 = load <4 x i16>, <4 x i16>* %A
	%tmp2 = load <4 x i16>, <4 x i16>* %B
	%tmp3 = call <4 x i16> @llvm.aarch64.neon.uhadd.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <8 x i16> @uhadd8h(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: uhadd8h:
;CHECK: uhadd.8h
	%tmp1 = load <8 x i16>, <8 x i16>* %A
	%tmp2 = load <8 x i16>, <8 x i16>* %B
	%tmp3 = call <8 x i16> @llvm.aarch64.neon.uhadd.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <2 x i32> @uhadd2s(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: uhadd2s:
;CHECK: uhadd.2s
	%tmp1 = load <2 x i32>, <2 x i32>* %A
	%tmp2 = load <2 x i32>, <2 x i32>* %B
	%tmp3 = call <2 x i32> @llvm.aarch64.neon.uhadd.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <4 x i32> @uhadd4s(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: uhadd4s:
;CHECK: uhadd.4s
	%tmp1 = load <4 x i32>, <4 x i32>* %A
	%tmp2 = load <4 x i32>, <4 x i32>* %B
	%tmp3 = call <4 x i32> @llvm.aarch64.neon.uhadd.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

declare <8 x i8>  @llvm.aarch64.neon.shadd.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.shadd.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.aarch64.neon.shadd.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

declare <8 x i8>  @llvm.aarch64.neon.uhadd.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.uhadd.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.aarch64.neon.uhadd.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

declare <16 x i8> @llvm.aarch64.neon.shadd.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.aarch64.neon.shadd.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.aarch64.neon.shadd.v4i32(<4 x i32>, <4 x i32>) nounwind readnone

declare <16 x i8> @llvm.aarch64.neon.uhadd.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.aarch64.neon.uhadd.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.aarch64.neon.uhadd.v4i32(<4 x i32>, <4 x i32>) nounwind readnone

define <8 x i8> @srhadd8b(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: srhadd8b:
;CHECK: srhadd.8b
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = load <8 x i8>, <8 x i8>* %B
	%tmp3 = call <8 x i8> @llvm.aarch64.neon.srhadd.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <16 x i8> @srhadd16b(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: srhadd16b:
;CHECK: srhadd.16b
	%tmp1 = load <16 x i8>, <16 x i8>* %A
	%tmp2 = load <16 x i8>, <16 x i8>* %B
	%tmp3 = call <16 x i8> @llvm.aarch64.neon.srhadd.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

define <4 x i16> @srhadd4h(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: srhadd4h:
;CHECK: srhadd.4h
	%tmp1 = load <4 x i16>, <4 x i16>* %A
	%tmp2 = load <4 x i16>, <4 x i16>* %B
	%tmp3 = call <4 x i16> @llvm.aarch64.neon.srhadd.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <8 x i16> @srhadd8h(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: srhadd8h:
;CHECK: srhadd.8h
	%tmp1 = load <8 x i16>, <8 x i16>* %A
	%tmp2 = load <8 x i16>, <8 x i16>* %B
	%tmp3 = call <8 x i16> @llvm.aarch64.neon.srhadd.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <2 x i32> @srhadd2s(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: srhadd2s:
;CHECK: srhadd.2s
	%tmp1 = load <2 x i32>, <2 x i32>* %A
	%tmp2 = load <2 x i32>, <2 x i32>* %B
	%tmp3 = call <2 x i32> @llvm.aarch64.neon.srhadd.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <4 x i32> @srhadd4s(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: srhadd4s:
;CHECK: srhadd.4s
	%tmp1 = load <4 x i32>, <4 x i32>* %A
	%tmp2 = load <4 x i32>, <4 x i32>* %B
	%tmp3 = call <4 x i32> @llvm.aarch64.neon.srhadd.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

define <8 x i8> @urhadd8b(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: urhadd8b:
;CHECK: urhadd.8b
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = load <8 x i8>, <8 x i8>* %B
	%tmp3 = call <8 x i8> @llvm.aarch64.neon.urhadd.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <16 x i8> @urhadd16b(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: urhadd16b:
;CHECK: urhadd.16b
	%tmp1 = load <16 x i8>, <16 x i8>* %A
	%tmp2 = load <16 x i8>, <16 x i8>* %B
	%tmp3 = call <16 x i8> @llvm.aarch64.neon.urhadd.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

define <4 x i16> @urhadd4h(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: urhadd4h:
;CHECK: urhadd.4h
	%tmp1 = load <4 x i16>, <4 x i16>* %A
	%tmp2 = load <4 x i16>, <4 x i16>* %B
	%tmp3 = call <4 x i16> @llvm.aarch64.neon.urhadd.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <8 x i16> @urhadd8h(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: urhadd8h:
;CHECK: urhadd.8h
	%tmp1 = load <8 x i16>, <8 x i16>* %A
	%tmp2 = load <8 x i16>, <8 x i16>* %B
	%tmp3 = call <8 x i16> @llvm.aarch64.neon.urhadd.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <2 x i32> @urhadd2s(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: urhadd2s:
;CHECK: urhadd.2s
	%tmp1 = load <2 x i32>, <2 x i32>* %A
	%tmp2 = load <2 x i32>, <2 x i32>* %B
	%tmp3 = call <2 x i32> @llvm.aarch64.neon.urhadd.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <4 x i32> @urhadd4s(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: urhadd4s:
;CHECK: urhadd.4s
	%tmp1 = load <4 x i32>, <4 x i32>* %A
	%tmp2 = load <4 x i32>, <4 x i32>* %B
	%tmp3 = call <4 x i32> @llvm.aarch64.neon.urhadd.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

declare <8 x i8>  @llvm.aarch64.neon.srhadd.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.srhadd.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.aarch64.neon.srhadd.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

declare <8 x i8>  @llvm.aarch64.neon.urhadd.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.urhadd.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.aarch64.neon.urhadd.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

declare <16 x i8> @llvm.aarch64.neon.srhadd.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.aarch64.neon.srhadd.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.aarch64.neon.srhadd.v4i32(<4 x i32>, <4 x i32>) nounwind readnone

declare <16 x i8> @llvm.aarch64.neon.urhadd.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.aarch64.neon.urhadd.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.aarch64.neon.urhadd.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
