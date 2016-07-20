; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple | FileCheck %s

define <8 x i8> @smax_8b(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: smax_8b:
;CHECK: smax.8b
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = load <8 x i8>, <8 x i8>* %B
	%tmp3 = call <8 x i8> @llvm.aarch64.neon.smax.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <16 x i8> @smax_16b(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: smax_16b:
;CHECK: smax.16b
	%tmp1 = load <16 x i8>, <16 x i8>* %A
	%tmp2 = load <16 x i8>, <16 x i8>* %B
	%tmp3 = call <16 x i8> @llvm.aarch64.neon.smax.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

define <4 x i16> @smax_4h(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: smax_4h:
;CHECK: smax.4h
	%tmp1 = load <4 x i16>, <4 x i16>* %A
	%tmp2 = load <4 x i16>, <4 x i16>* %B
	%tmp3 = call <4 x i16> @llvm.aarch64.neon.smax.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <8 x i16> @smax_8h(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: smax_8h:
;CHECK: smax.8h
	%tmp1 = load <8 x i16>, <8 x i16>* %A
	%tmp2 = load <8 x i16>, <8 x i16>* %B
	%tmp3 = call <8 x i16> @llvm.aarch64.neon.smax.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <2 x i32> @smax_2s(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: smax_2s:
;CHECK: smax.2s
	%tmp1 = load <2 x i32>, <2 x i32>* %A
	%tmp2 = load <2 x i32>, <2 x i32>* %B
	%tmp3 = call <2 x i32> @llvm.aarch64.neon.smax.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <4 x i32> @smax_4s(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: smax_4s:
;CHECK: smax.4s
	%tmp1 = load <4 x i32>, <4 x i32>* %A
	%tmp2 = load <4 x i32>, <4 x i32>* %B
	%tmp3 = call <4 x i32> @llvm.aarch64.neon.smax.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

declare <8 x i8> @llvm.aarch64.neon.smax.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <16 x i8> @llvm.aarch64.neon.smax.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.smax.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <8 x i16> @llvm.aarch64.neon.smax.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <2 x i32> @llvm.aarch64.neon.smax.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <4 x i32> @llvm.aarch64.neon.smax.v4i32(<4 x i32>, <4 x i32>) nounwind readnone

define <8 x i8> @umax_8b(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: umax_8b:
;CHECK: umax.8b
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = load <8 x i8>, <8 x i8>* %B
	%tmp3 = call <8 x i8> @llvm.aarch64.neon.umax.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <16 x i8> @umax_16b(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: umax_16b:
;CHECK: umax.16b
	%tmp1 = load <16 x i8>, <16 x i8>* %A
	%tmp2 = load <16 x i8>, <16 x i8>* %B
	%tmp3 = call <16 x i8> @llvm.aarch64.neon.umax.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

define <4 x i16> @umax_4h(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: umax_4h:
;CHECK: umax.4h
	%tmp1 = load <4 x i16>, <4 x i16>* %A
	%tmp2 = load <4 x i16>, <4 x i16>* %B
	%tmp3 = call <4 x i16> @llvm.aarch64.neon.umax.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <8 x i16> @umax_8h(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: umax_8h:
;CHECK: umax.8h
	%tmp1 = load <8 x i16>, <8 x i16>* %A
	%tmp2 = load <8 x i16>, <8 x i16>* %B
	%tmp3 = call <8 x i16> @llvm.aarch64.neon.umax.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <2 x i32> @umax_2s(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: umax_2s:
;CHECK: umax.2s
	%tmp1 = load <2 x i32>, <2 x i32>* %A
	%tmp2 = load <2 x i32>, <2 x i32>* %B
	%tmp3 = call <2 x i32> @llvm.aarch64.neon.umax.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <4 x i32> @umax_4s(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: umax_4s:
;CHECK: umax.4s
	%tmp1 = load <4 x i32>, <4 x i32>* %A
	%tmp2 = load <4 x i32>, <4 x i32>* %B
	%tmp3 = call <4 x i32> @llvm.aarch64.neon.umax.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

declare <8 x i8> @llvm.aarch64.neon.umax.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <16 x i8> @llvm.aarch64.neon.umax.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.umax.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <8 x i16> @llvm.aarch64.neon.umax.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <2 x i32> @llvm.aarch64.neon.umax.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <4 x i32> @llvm.aarch64.neon.umax.v4i32(<4 x i32>, <4 x i32>) nounwind readnone

define <8 x i8> @smin_8b(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: smin_8b:
;CHECK: smin.8b
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = load <8 x i8>, <8 x i8>* %B
	%tmp3 = call <8 x i8> @llvm.aarch64.neon.smin.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <16 x i8> @smin_16b(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: smin_16b:
;CHECK: smin.16b
	%tmp1 = load <16 x i8>, <16 x i8>* %A
	%tmp2 = load <16 x i8>, <16 x i8>* %B
	%tmp3 = call <16 x i8> @llvm.aarch64.neon.smin.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

define <4 x i16> @smin_4h(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: smin_4h:
;CHECK: smin.4h
	%tmp1 = load <4 x i16>, <4 x i16>* %A
	%tmp2 = load <4 x i16>, <4 x i16>* %B
	%tmp3 = call <4 x i16> @llvm.aarch64.neon.smin.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <8 x i16> @smin_8h(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: smin_8h:
;CHECK: smin.8h
	%tmp1 = load <8 x i16>, <8 x i16>* %A
	%tmp2 = load <8 x i16>, <8 x i16>* %B
	%tmp3 = call <8 x i16> @llvm.aarch64.neon.smin.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <2 x i32> @smin_2s(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: smin_2s:
;CHECK: smin.2s
	%tmp1 = load <2 x i32>, <2 x i32>* %A
	%tmp2 = load <2 x i32>, <2 x i32>* %B
	%tmp3 = call <2 x i32> @llvm.aarch64.neon.smin.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <4 x i32> @smin_4s(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: smin_4s:
;CHECK: smin.4s
	%tmp1 = load <4 x i32>, <4 x i32>* %A
	%tmp2 = load <4 x i32>, <4 x i32>* %B
	%tmp3 = call <4 x i32> @llvm.aarch64.neon.smin.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

declare <8 x i8> @llvm.aarch64.neon.smin.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <16 x i8> @llvm.aarch64.neon.smin.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.smin.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <8 x i16> @llvm.aarch64.neon.smin.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <2 x i32> @llvm.aarch64.neon.smin.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <4 x i32> @llvm.aarch64.neon.smin.v4i32(<4 x i32>, <4 x i32>) nounwind readnone

define <8 x i8> @umin_8b(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: umin_8b:
;CHECK: umin.8b
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = load <8 x i8>, <8 x i8>* %B
	%tmp3 = call <8 x i8> @llvm.aarch64.neon.umin.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <16 x i8> @umin_16b(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: umin_16b:
;CHECK: umin.16b
	%tmp1 = load <16 x i8>, <16 x i8>* %A
	%tmp2 = load <16 x i8>, <16 x i8>* %B
	%tmp3 = call <16 x i8> @llvm.aarch64.neon.umin.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

define <4 x i16> @umin_4h(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: umin_4h:
;CHECK: umin.4h
	%tmp1 = load <4 x i16>, <4 x i16>* %A
	%tmp2 = load <4 x i16>, <4 x i16>* %B
	%tmp3 = call <4 x i16> @llvm.aarch64.neon.umin.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <8 x i16> @umin_8h(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: umin_8h:
;CHECK: umin.8h
	%tmp1 = load <8 x i16>, <8 x i16>* %A
	%tmp2 = load <8 x i16>, <8 x i16>* %B
	%tmp3 = call <8 x i16> @llvm.aarch64.neon.umin.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <2 x i32> @umin_2s(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: umin_2s:
;CHECK: umin.2s
	%tmp1 = load <2 x i32>, <2 x i32>* %A
	%tmp2 = load <2 x i32>, <2 x i32>* %B
	%tmp3 = call <2 x i32> @llvm.aarch64.neon.umin.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <4 x i32> @umin_4s(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: umin_4s:
;CHECK: umin.4s
	%tmp1 = load <4 x i32>, <4 x i32>* %A
	%tmp2 = load <4 x i32>, <4 x i32>* %B
	%tmp3 = call <4 x i32> @llvm.aarch64.neon.umin.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

declare <8 x i8> @llvm.aarch64.neon.umin.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <16 x i8> @llvm.aarch64.neon.umin.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.umin.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <8 x i16> @llvm.aarch64.neon.umin.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <2 x i32> @llvm.aarch64.neon.umin.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <4 x i32> @llvm.aarch64.neon.umin.v4i32(<4 x i32>, <4 x i32>) nounwind readnone

; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple | FileCheck %s

define <8 x i8> @smaxp_8b(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: smaxp_8b:
;CHECK: smaxp.8b
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = load <8 x i8>, <8 x i8>* %B
	%tmp3 = call <8 x i8> @llvm.aarch64.neon.smaxp.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <16 x i8> @smaxp_16b(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: smaxp_16b:
;CHECK: smaxp.16b
	%tmp1 = load <16 x i8>, <16 x i8>* %A
	%tmp2 = load <16 x i8>, <16 x i8>* %B
	%tmp3 = call <16 x i8> @llvm.aarch64.neon.smaxp.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

define <4 x i16> @smaxp_4h(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: smaxp_4h:
;CHECK: smaxp.4h
	%tmp1 = load <4 x i16>, <4 x i16>* %A
	%tmp2 = load <4 x i16>, <4 x i16>* %B
	%tmp3 = call <4 x i16> @llvm.aarch64.neon.smaxp.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <8 x i16> @smaxp_8h(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: smaxp_8h:
;CHECK: smaxp.8h
	%tmp1 = load <8 x i16>, <8 x i16>* %A
	%tmp2 = load <8 x i16>, <8 x i16>* %B
	%tmp3 = call <8 x i16> @llvm.aarch64.neon.smaxp.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <2 x i32> @smaxp_2s(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: smaxp_2s:
;CHECK: smaxp.2s
	%tmp1 = load <2 x i32>, <2 x i32>* %A
	%tmp2 = load <2 x i32>, <2 x i32>* %B
	%tmp3 = call <2 x i32> @llvm.aarch64.neon.smaxp.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <4 x i32> @smaxp_4s(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: smaxp_4s:
;CHECK: smaxp.4s
	%tmp1 = load <4 x i32>, <4 x i32>* %A
	%tmp2 = load <4 x i32>, <4 x i32>* %B
	%tmp3 = call <4 x i32> @llvm.aarch64.neon.smaxp.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

declare <8 x i8> @llvm.aarch64.neon.smaxp.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <16 x i8> @llvm.aarch64.neon.smaxp.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.smaxp.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <8 x i16> @llvm.aarch64.neon.smaxp.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <2 x i32> @llvm.aarch64.neon.smaxp.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <4 x i32> @llvm.aarch64.neon.smaxp.v4i32(<4 x i32>, <4 x i32>) nounwind readnone

define <8 x i8> @umaxp_8b(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: umaxp_8b:
;CHECK: umaxp.8b
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = load <8 x i8>, <8 x i8>* %B
	%tmp3 = call <8 x i8> @llvm.aarch64.neon.umaxp.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <16 x i8> @umaxp_16b(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: umaxp_16b:
;CHECK: umaxp.16b
	%tmp1 = load <16 x i8>, <16 x i8>* %A
	%tmp2 = load <16 x i8>, <16 x i8>* %B
	%tmp3 = call <16 x i8> @llvm.aarch64.neon.umaxp.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

define <4 x i16> @umaxp_4h(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: umaxp_4h:
;CHECK: umaxp.4h
	%tmp1 = load <4 x i16>, <4 x i16>* %A
	%tmp2 = load <4 x i16>, <4 x i16>* %B
	%tmp3 = call <4 x i16> @llvm.aarch64.neon.umaxp.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <8 x i16> @umaxp_8h(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: umaxp_8h:
;CHECK: umaxp.8h
	%tmp1 = load <8 x i16>, <8 x i16>* %A
	%tmp2 = load <8 x i16>, <8 x i16>* %B
	%tmp3 = call <8 x i16> @llvm.aarch64.neon.umaxp.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <2 x i32> @umaxp_2s(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: umaxp_2s:
;CHECK: umaxp.2s
	%tmp1 = load <2 x i32>, <2 x i32>* %A
	%tmp2 = load <2 x i32>, <2 x i32>* %B
	%tmp3 = call <2 x i32> @llvm.aarch64.neon.umaxp.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <4 x i32> @umaxp_4s(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: umaxp_4s:
;CHECK: umaxp.4s
	%tmp1 = load <4 x i32>, <4 x i32>* %A
	%tmp2 = load <4 x i32>, <4 x i32>* %B
	%tmp3 = call <4 x i32> @llvm.aarch64.neon.umaxp.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

declare <8 x i8> @llvm.aarch64.neon.umaxp.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <16 x i8> @llvm.aarch64.neon.umaxp.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.umaxp.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <8 x i16> @llvm.aarch64.neon.umaxp.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <2 x i32> @llvm.aarch64.neon.umaxp.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <4 x i32> @llvm.aarch64.neon.umaxp.v4i32(<4 x i32>, <4 x i32>) nounwind readnone

; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple | FileCheck %s

define <8 x i8> @sminp_8b(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: sminp_8b:
;CHECK: sminp.8b
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = load <8 x i8>, <8 x i8>* %B
	%tmp3 = call <8 x i8> @llvm.aarch64.neon.sminp.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <16 x i8> @sminp_16b(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: sminp_16b:
;CHECK: sminp.16b
	%tmp1 = load <16 x i8>, <16 x i8>* %A
	%tmp2 = load <16 x i8>, <16 x i8>* %B
	%tmp3 = call <16 x i8> @llvm.aarch64.neon.sminp.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

define <4 x i16> @sminp_4h(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: sminp_4h:
;CHECK: sminp.4h
	%tmp1 = load <4 x i16>, <4 x i16>* %A
	%tmp2 = load <4 x i16>, <4 x i16>* %B
	%tmp3 = call <4 x i16> @llvm.aarch64.neon.sminp.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <8 x i16> @sminp_8h(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: sminp_8h:
;CHECK: sminp.8h
	%tmp1 = load <8 x i16>, <8 x i16>* %A
	%tmp2 = load <8 x i16>, <8 x i16>* %B
	%tmp3 = call <8 x i16> @llvm.aarch64.neon.sminp.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <2 x i32> @sminp_2s(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: sminp_2s:
;CHECK: sminp.2s
	%tmp1 = load <2 x i32>, <2 x i32>* %A
	%tmp2 = load <2 x i32>, <2 x i32>* %B
	%tmp3 = call <2 x i32> @llvm.aarch64.neon.sminp.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <4 x i32> @sminp_4s(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: sminp_4s:
;CHECK: sminp.4s
	%tmp1 = load <4 x i32>, <4 x i32>* %A
	%tmp2 = load <4 x i32>, <4 x i32>* %B
	%tmp3 = call <4 x i32> @llvm.aarch64.neon.sminp.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

declare <8 x i8> @llvm.aarch64.neon.sminp.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <16 x i8> @llvm.aarch64.neon.sminp.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.sminp.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <8 x i16> @llvm.aarch64.neon.sminp.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <2 x i32> @llvm.aarch64.neon.sminp.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <4 x i32> @llvm.aarch64.neon.sminp.v4i32(<4 x i32>, <4 x i32>) nounwind readnone

define <8 x i8> @uminp_8b(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: uminp_8b:
;CHECK: uminp.8b
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = load <8 x i8>, <8 x i8>* %B
	%tmp3 = call <8 x i8> @llvm.aarch64.neon.uminp.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <16 x i8> @uminp_16b(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: uminp_16b:
;CHECK: uminp.16b
	%tmp1 = load <16 x i8>, <16 x i8>* %A
	%tmp2 = load <16 x i8>, <16 x i8>* %B
	%tmp3 = call <16 x i8> @llvm.aarch64.neon.uminp.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

define <4 x i16> @uminp_4h(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: uminp_4h:
;CHECK: uminp.4h
	%tmp1 = load <4 x i16>, <4 x i16>* %A
	%tmp2 = load <4 x i16>, <4 x i16>* %B
	%tmp3 = call <4 x i16> @llvm.aarch64.neon.uminp.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <8 x i16> @uminp_8h(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: uminp_8h:
;CHECK: uminp.8h
	%tmp1 = load <8 x i16>, <8 x i16>* %A
	%tmp2 = load <8 x i16>, <8 x i16>* %B
	%tmp3 = call <8 x i16> @llvm.aarch64.neon.uminp.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <2 x i32> @uminp_2s(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: uminp_2s:
;CHECK: uminp.2s
	%tmp1 = load <2 x i32>, <2 x i32>* %A
	%tmp2 = load <2 x i32>, <2 x i32>* %B
	%tmp3 = call <2 x i32> @llvm.aarch64.neon.uminp.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <4 x i32> @uminp_4s(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: uminp_4s:
;CHECK: uminp.4s
	%tmp1 = load <4 x i32>, <4 x i32>* %A
	%tmp2 = load <4 x i32>, <4 x i32>* %B
	%tmp3 = call <4 x i32> @llvm.aarch64.neon.uminp.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

declare <8 x i8> @llvm.aarch64.neon.uminp.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <16 x i8> @llvm.aarch64.neon.uminp.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.uminp.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <8 x i16> @llvm.aarch64.neon.uminp.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <2 x i32> @llvm.aarch64.neon.uminp.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <4 x i32> @llvm.aarch64.neon.uminp.v4i32(<4 x i32>, <4 x i32>) nounwind readnone

define <2 x float> @fmax_2s(<2 x float>* %A, <2 x float>* %B) nounwind {
;CHECK-LABEL: fmax_2s:
;CHECK: fmax.2s
	%tmp1 = load <2 x float>, <2 x float>* %A
	%tmp2 = load <2 x float>, <2 x float>* %B
	%tmp3 = call <2 x float> @llvm.aarch64.neon.fmax.v2f32(<2 x float> %tmp1, <2 x float> %tmp2)
	ret <2 x float> %tmp3
}

define <4 x float> @fmax_4s(<4 x float>* %A, <4 x float>* %B) nounwind {
;CHECK-LABEL: fmax_4s:
;CHECK: fmax.4s
	%tmp1 = load <4 x float>, <4 x float>* %A
	%tmp2 = load <4 x float>, <4 x float>* %B
	%tmp3 = call <4 x float> @llvm.aarch64.neon.fmax.v4f32(<4 x float> %tmp1, <4 x float> %tmp2)
	ret <4 x float> %tmp3
}

define <2 x double> @fmax_2d(<2 x double>* %A, <2 x double>* %B) nounwind {
;CHECK-LABEL: fmax_2d:
;CHECK: fmax.2d
	%tmp1 = load <2 x double>, <2 x double>* %A
	%tmp2 = load <2 x double>, <2 x double>* %B
	%tmp3 = call <2 x double> @llvm.aarch64.neon.fmax.v2f64(<2 x double> %tmp1, <2 x double> %tmp2)
	ret <2 x double> %tmp3
}

declare <2 x float> @llvm.aarch64.neon.fmax.v2f32(<2 x float>, <2 x float>) nounwind readnone
declare <4 x float> @llvm.aarch64.neon.fmax.v4f32(<4 x float>, <4 x float>) nounwind readnone
declare <2 x double> @llvm.aarch64.neon.fmax.v2f64(<2 x double>, <2 x double>) nounwind readnone

define <2 x float> @fmaxp_2s(<2 x float>* %A, <2 x float>* %B) nounwind {
;CHECK-LABEL: fmaxp_2s:
;CHECK: fmaxp.2s
	%tmp1 = load <2 x float>, <2 x float>* %A
	%tmp2 = load <2 x float>, <2 x float>* %B
	%tmp3 = call <2 x float> @llvm.aarch64.neon.fmaxp.v2f32(<2 x float> %tmp1, <2 x float> %tmp2)
	ret <2 x float> %tmp3
}

define <4 x float> @fmaxp_4s(<4 x float>* %A, <4 x float>* %B) nounwind {
;CHECK-LABEL: fmaxp_4s:
;CHECK: fmaxp.4s
	%tmp1 = load <4 x float>, <4 x float>* %A
	%tmp2 = load <4 x float>, <4 x float>* %B
	%tmp3 = call <4 x float> @llvm.aarch64.neon.fmaxp.v4f32(<4 x float> %tmp1, <4 x float> %tmp2)
	ret <4 x float> %tmp3
}

define <2 x double> @fmaxp_2d(<2 x double>* %A, <2 x double>* %B) nounwind {
;CHECK-LABEL: fmaxp_2d:
;CHECK: fmaxp.2d
	%tmp1 = load <2 x double>, <2 x double>* %A
	%tmp2 = load <2 x double>, <2 x double>* %B
	%tmp3 = call <2 x double> @llvm.aarch64.neon.fmaxp.v2f64(<2 x double> %tmp1, <2 x double> %tmp2)
	ret <2 x double> %tmp3
}

declare <2 x float> @llvm.aarch64.neon.fmaxp.v2f32(<2 x float>, <2 x float>) nounwind readnone
declare <4 x float> @llvm.aarch64.neon.fmaxp.v4f32(<4 x float>, <4 x float>) nounwind readnone
declare <2 x double> @llvm.aarch64.neon.fmaxp.v2f64(<2 x double>, <2 x double>) nounwind readnone

define <2 x float> @fmin_2s(<2 x float>* %A, <2 x float>* %B) nounwind {
;CHECK-LABEL: fmin_2s:
;CHECK: fmin.2s
	%tmp1 = load <2 x float>, <2 x float>* %A
	%tmp2 = load <2 x float>, <2 x float>* %B
	%tmp3 = call <2 x float> @llvm.aarch64.neon.fmin.v2f32(<2 x float> %tmp1, <2 x float> %tmp2)
	ret <2 x float> %tmp3
}

define <4 x float> @fmin_4s(<4 x float>* %A, <4 x float>* %B) nounwind {
;CHECK-LABEL: fmin_4s:
;CHECK: fmin.4s
	%tmp1 = load <4 x float>, <4 x float>* %A
	%tmp2 = load <4 x float>, <4 x float>* %B
	%tmp3 = call <4 x float> @llvm.aarch64.neon.fmin.v4f32(<4 x float> %tmp1, <4 x float> %tmp2)
	ret <4 x float> %tmp3
}

define <2 x double> @fmin_2d(<2 x double>* %A, <2 x double>* %B) nounwind {
;CHECK-LABEL: fmin_2d:
;CHECK: fmin.2d
	%tmp1 = load <2 x double>, <2 x double>* %A
	%tmp2 = load <2 x double>, <2 x double>* %B
	%tmp3 = call <2 x double> @llvm.aarch64.neon.fmin.v2f64(<2 x double> %tmp1, <2 x double> %tmp2)
	ret <2 x double> %tmp3
}

declare <2 x float> @llvm.aarch64.neon.fmin.v2f32(<2 x float>, <2 x float>) nounwind readnone
declare <4 x float> @llvm.aarch64.neon.fmin.v4f32(<4 x float>, <4 x float>) nounwind readnone
declare <2 x double> @llvm.aarch64.neon.fmin.v2f64(<2 x double>, <2 x double>) nounwind readnone

define <2 x float> @fminp_2s(<2 x float>* %A, <2 x float>* %B) nounwind {
;CHECK-LABEL: fminp_2s:
;CHECK: fminp.2s
	%tmp1 = load <2 x float>, <2 x float>* %A
	%tmp2 = load <2 x float>, <2 x float>* %B
	%tmp3 = call <2 x float> @llvm.aarch64.neon.fminp.v2f32(<2 x float> %tmp1, <2 x float> %tmp2)
	ret <2 x float> %tmp3
}

define <4 x float> @fminp_4s(<4 x float>* %A, <4 x float>* %B) nounwind {
;CHECK-LABEL: fminp_4s:
;CHECK: fminp.4s
	%tmp1 = load <4 x float>, <4 x float>* %A
	%tmp2 = load <4 x float>, <4 x float>* %B
	%tmp3 = call <4 x float> @llvm.aarch64.neon.fminp.v4f32(<4 x float> %tmp1, <4 x float> %tmp2)
	ret <4 x float> %tmp3
}

define <2 x double> @fminp_2d(<2 x double>* %A, <2 x double>* %B) nounwind {
;CHECK-LABEL: fminp_2d:
;CHECK: fminp.2d
	%tmp1 = load <2 x double>, <2 x double>* %A
	%tmp2 = load <2 x double>, <2 x double>* %B
	%tmp3 = call <2 x double> @llvm.aarch64.neon.fminp.v2f64(<2 x double> %tmp1, <2 x double> %tmp2)
	ret <2 x double> %tmp3
}

declare <2 x float> @llvm.aarch64.neon.fminp.v2f32(<2 x float>, <2 x float>) nounwind readnone
declare <4 x float> @llvm.aarch64.neon.fminp.v4f32(<4 x float>, <4 x float>) nounwind readnone
declare <2 x double> @llvm.aarch64.neon.fminp.v2f64(<2 x double>, <2 x double>) nounwind readnone

define <2 x float> @fminnmp_2s(<2 x float>* %A, <2 x float>* %B) nounwind {
;CHECK-LABEL: fminnmp_2s:
;CHECK: fminnmp.2s
	%tmp1 = load <2 x float>, <2 x float>* %A
	%tmp2 = load <2 x float>, <2 x float>* %B
	%tmp3 = call <2 x float> @llvm.aarch64.neon.fminnmp.v2f32(<2 x float> %tmp1, <2 x float> %tmp2)
	ret <2 x float> %tmp3
}

define <4 x float> @fminnmp_4s(<4 x float>* %A, <4 x float>* %B) nounwind {
;CHECK-LABEL: fminnmp_4s:
;CHECK: fminnmp.4s
	%tmp1 = load <4 x float>, <4 x float>* %A
	%tmp2 = load <4 x float>, <4 x float>* %B
	%tmp3 = call <4 x float> @llvm.aarch64.neon.fminnmp.v4f32(<4 x float> %tmp1, <4 x float> %tmp2)
	ret <4 x float> %tmp3
}

define <2 x double> @fminnmp_2d(<2 x double>* %A, <2 x double>* %B) nounwind {
;CHECK-LABEL: fminnmp_2d:
;CHECK: fminnmp.2d
	%tmp1 = load <2 x double>, <2 x double>* %A
	%tmp2 = load <2 x double>, <2 x double>* %B
	%tmp3 = call <2 x double> @llvm.aarch64.neon.fminnmp.v2f64(<2 x double> %tmp1, <2 x double> %tmp2)
	ret <2 x double> %tmp3
}

declare <2 x float> @llvm.aarch64.neon.fminnmp.v2f32(<2 x float>, <2 x float>) nounwind readnone
declare <4 x float> @llvm.aarch64.neon.fminnmp.v4f32(<4 x float>, <4 x float>) nounwind readnone
declare <2 x double> @llvm.aarch64.neon.fminnmp.v2f64(<2 x double>, <2 x double>) nounwind readnone

define <2 x float> @fmaxnmp_2s(<2 x float>* %A, <2 x float>* %B) nounwind {
;CHECK-LABEL: fmaxnmp_2s:
;CHECK: fmaxnmp.2s
	%tmp1 = load <2 x float>, <2 x float>* %A
	%tmp2 = load <2 x float>, <2 x float>* %B
	%tmp3 = call <2 x float> @llvm.aarch64.neon.fmaxnmp.v2f32(<2 x float> %tmp1, <2 x float> %tmp2)
	ret <2 x float> %tmp3
}

define <4 x float> @fmaxnmp_4s(<4 x float>* %A, <4 x float>* %B) nounwind {
;CHECK-LABEL: fmaxnmp_4s:
;CHECK: fmaxnmp.4s
	%tmp1 = load <4 x float>, <4 x float>* %A
	%tmp2 = load <4 x float>, <4 x float>* %B
	%tmp3 = call <4 x float> @llvm.aarch64.neon.fmaxnmp.v4f32(<4 x float> %tmp1, <4 x float> %tmp2)
	ret <4 x float> %tmp3
}

define <2 x double> @fmaxnmp_2d(<2 x double>* %A, <2 x double>* %B) nounwind {
;CHECK-LABEL: fmaxnmp_2d:
;CHECK: fmaxnmp.2d
	%tmp1 = load <2 x double>, <2 x double>* %A
	%tmp2 = load <2 x double>, <2 x double>* %B
	%tmp3 = call <2 x double> @llvm.aarch64.neon.fmaxnmp.v2f64(<2 x double> %tmp1, <2 x double> %tmp2)
	ret <2 x double> %tmp3
}

declare <2 x float> @llvm.aarch64.neon.fmaxnmp.v2f32(<2 x float>, <2 x float>) nounwind readnone
declare <4 x float> @llvm.aarch64.neon.fmaxnmp.v4f32(<4 x float>, <4 x float>) nounwind readnone
declare <2 x double> @llvm.aarch64.neon.fmaxnmp.v2f64(<2 x double>, <2 x double>) nounwind readnone
