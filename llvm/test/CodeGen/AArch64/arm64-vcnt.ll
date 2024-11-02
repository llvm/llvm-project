; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple | FileCheck %s

define <8 x i8> @cls_8b(ptr %A) nounwind {
;CHECK-LABEL: cls_8b:
;CHECK: cls.8b
	%tmp1 = load <8 x i8>, ptr %A
	%tmp3 = call <8 x i8> @llvm.aarch64.neon.cls.v8i8(<8 x i8> %tmp1)
	ret <8 x i8> %tmp3
}

define <16 x i8> @cls_16b(ptr %A) nounwind {
;CHECK-LABEL: cls_16b:
;CHECK: cls.16b
	%tmp1 = load <16 x i8>, ptr %A
	%tmp3 = call <16 x i8> @llvm.aarch64.neon.cls.v16i8(<16 x i8> %tmp1)
	ret <16 x i8> %tmp3
}

define <4 x i16> @cls_4h(ptr %A) nounwind {
;CHECK-LABEL: cls_4h:
;CHECK: cls.4h
	%tmp1 = load <4 x i16>, ptr %A
	%tmp3 = call <4 x i16> @llvm.aarch64.neon.cls.v4i16(<4 x i16> %tmp1)
	ret <4 x i16> %tmp3
}

define <8 x i16> @cls_8h(ptr %A) nounwind {
;CHECK-LABEL: cls_8h:
;CHECK: cls.8h
	%tmp1 = load <8 x i16>, ptr %A
	%tmp3 = call <8 x i16> @llvm.aarch64.neon.cls.v8i16(<8 x i16> %tmp1)
	ret <8 x i16> %tmp3
}

define <2 x i32> @cls_2s(ptr %A) nounwind {
;CHECK-LABEL: cls_2s:
;CHECK: cls.2s
	%tmp1 = load <2 x i32>, ptr %A
	%tmp3 = call <2 x i32> @llvm.aarch64.neon.cls.v2i32(<2 x i32> %tmp1)
	ret <2 x i32> %tmp3
}

define <4 x i32> @cls_4s(ptr %A) nounwind {
;CHECK-LABEL: cls_4s:
;CHECK: cls.4s
	%tmp1 = load <4 x i32>, ptr %A
	%tmp3 = call <4 x i32> @llvm.aarch64.neon.cls.v4i32(<4 x i32> %tmp1)
	ret <4 x i32> %tmp3
}

declare <8 x i8> @llvm.aarch64.neon.cls.v8i8(<8 x i8>) nounwind readnone
declare <16 x i8> @llvm.aarch64.neon.cls.v16i8(<16 x i8>) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.cls.v4i16(<4 x i16>) nounwind readnone
declare <8 x i16> @llvm.aarch64.neon.cls.v8i16(<8 x i16>) nounwind readnone
declare <2 x i32> @llvm.aarch64.neon.cls.v2i32(<2 x i32>) nounwind readnone
declare <4 x i32> @llvm.aarch64.neon.cls.v4i32(<4 x i32>) nounwind readnone
