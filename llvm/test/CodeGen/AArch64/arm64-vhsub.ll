; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple | FileCheck %s

define <8 x i8> @shsub8b(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: shsub8b:
;CHECK: shsub.8b
	%tmp1 = load <8 x i8>, ptr %A
	%tmp2 = load <8 x i8>, ptr %B
	%tmp3 = call <8 x i8> @llvm.aarch64.neon.shsub.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <16 x i8> @shsub16b(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: shsub16b:
;CHECK: shsub.16b
	%tmp1 = load <16 x i8>, ptr %A
	%tmp2 = load <16 x i8>, ptr %B
	%tmp3 = call <16 x i8> @llvm.aarch64.neon.shsub.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

define <4 x i16> @shsub4h(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: shsub4h:
;CHECK: shsub.4h
	%tmp1 = load <4 x i16>, ptr %A
	%tmp2 = load <4 x i16>, ptr %B
	%tmp3 = call <4 x i16> @llvm.aarch64.neon.shsub.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <8 x i16> @shsub8h(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: shsub8h:
;CHECK: shsub.8h
	%tmp1 = load <8 x i16>, ptr %A
	%tmp2 = load <8 x i16>, ptr %B
	%tmp3 = call <8 x i16> @llvm.aarch64.neon.shsub.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <2 x i32> @shsub2s(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: shsub2s:
;CHECK: shsub.2s
	%tmp1 = load <2 x i32>, ptr %A
	%tmp2 = load <2 x i32>, ptr %B
	%tmp3 = call <2 x i32> @llvm.aarch64.neon.shsub.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <4 x i32> @shsub4s(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: shsub4s:
;CHECK: shsub.4s
	%tmp1 = load <4 x i32>, ptr %A
	%tmp2 = load <4 x i32>, ptr %B
	%tmp3 = call <4 x i32> @llvm.aarch64.neon.shsub.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

define <8 x i8> @uhsub8b(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: uhsub8b:
;CHECK: uhsub.8b
	%tmp1 = load <8 x i8>, ptr %A
	%tmp2 = load <8 x i8>, ptr %B
	%tmp3 = call <8 x i8> @llvm.aarch64.neon.uhsub.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <16 x i8> @uhsub16b(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: uhsub16b:
;CHECK: uhsub.16b
	%tmp1 = load <16 x i8>, ptr %A
	%tmp2 = load <16 x i8>, ptr %B
	%tmp3 = call <16 x i8> @llvm.aarch64.neon.uhsub.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

define <4 x i16> @uhsub4h(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: uhsub4h:
;CHECK: uhsub.4h
	%tmp1 = load <4 x i16>, ptr %A
	%tmp2 = load <4 x i16>, ptr %B
	%tmp3 = call <4 x i16> @llvm.aarch64.neon.uhsub.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <8 x i16> @uhsub8h(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: uhsub8h:
;CHECK: uhsub.8h
	%tmp1 = load <8 x i16>, ptr %A
	%tmp2 = load <8 x i16>, ptr %B
	%tmp3 = call <8 x i16> @llvm.aarch64.neon.uhsub.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <2 x i32> @uhsub2s(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: uhsub2s:
;CHECK: uhsub.2s
	%tmp1 = load <2 x i32>, ptr %A
	%tmp2 = load <2 x i32>, ptr %B
	%tmp3 = call <2 x i32> @llvm.aarch64.neon.uhsub.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <4 x i32> @uhsub4s(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: uhsub4s:
;CHECK: uhsub.4s
	%tmp1 = load <4 x i32>, ptr %A
	%tmp2 = load <4 x i32>, ptr %B
	%tmp3 = call <4 x i32> @llvm.aarch64.neon.uhsub.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

declare <8 x i8>  @llvm.aarch64.neon.shsub.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.shsub.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.aarch64.neon.shsub.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

declare <8 x i8>  @llvm.aarch64.neon.uhsub.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.uhsub.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.aarch64.neon.uhsub.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

declare <16 x i8> @llvm.aarch64.neon.shsub.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.aarch64.neon.shsub.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.aarch64.neon.shsub.v4i32(<4 x i32>, <4 x i32>) nounwind readnone

declare <16 x i8> @llvm.aarch64.neon.uhsub.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.aarch64.neon.uhsub.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.aarch64.neon.uhsub.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
