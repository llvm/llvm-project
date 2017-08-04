; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple -enable-misched=false | FileCheck %s

define <8 x i8> @sqshl8b(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: sqshl8b:
;CHECK: sqshl.8b
        %tmp1 = load <8 x i8>, <8 x i8>* %A
        %tmp2 = load <8 x i8>, <8 x i8>* %B
        %tmp3 = call <8 x i8> @llvm.aarch64.neon.sqshl.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
        ret <8 x i8> %tmp3
}

define <4 x i16> @sqshl4h(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: sqshl4h:
;CHECK: sqshl.4h
        %tmp1 = load <4 x i16>, <4 x i16>* %A
        %tmp2 = load <4 x i16>, <4 x i16>* %B
        %tmp3 = call <4 x i16> @llvm.aarch64.neon.sqshl.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
        ret <4 x i16> %tmp3
}

define <2 x i32> @sqshl2s(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: sqshl2s:
;CHECK: sqshl.2s
        %tmp1 = load <2 x i32>, <2 x i32>* %A
        %tmp2 = load <2 x i32>, <2 x i32>* %B
        %tmp3 = call <2 x i32> @llvm.aarch64.neon.sqshl.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
        ret <2 x i32> %tmp3
}

define <8 x i8> @uqshl8b(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: uqshl8b:
;CHECK: uqshl.8b
        %tmp1 = load <8 x i8>, <8 x i8>* %A
        %tmp2 = load <8 x i8>, <8 x i8>* %B
        %tmp3 = call <8 x i8> @llvm.aarch64.neon.uqshl.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
        ret <8 x i8> %tmp3
}

define <4 x i16> @uqshl4h(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: uqshl4h:
;CHECK: uqshl.4h
        %tmp1 = load <4 x i16>, <4 x i16>* %A
        %tmp2 = load <4 x i16>, <4 x i16>* %B
        %tmp3 = call <4 x i16> @llvm.aarch64.neon.uqshl.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
        ret <4 x i16> %tmp3
}

define <2 x i32> @uqshl2s(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: uqshl2s:
;CHECK: uqshl.2s
        %tmp1 = load <2 x i32>, <2 x i32>* %A
        %tmp2 = load <2 x i32>, <2 x i32>* %B
        %tmp3 = call <2 x i32> @llvm.aarch64.neon.uqshl.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
        ret <2 x i32> %tmp3
}

define <16 x i8> @sqshl16b(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: sqshl16b:
;CHECK: sqshl.16b
        %tmp1 = load <16 x i8>, <16 x i8>* %A
        %tmp2 = load <16 x i8>, <16 x i8>* %B
        %tmp3 = call <16 x i8> @llvm.aarch64.neon.sqshl.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
        ret <16 x i8> %tmp3
}

define <8 x i16> @sqshl8h(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: sqshl8h:
;CHECK: sqshl.8h
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp2 = load <8 x i16>, <8 x i16>* %B
        %tmp3 = call <8 x i16> @llvm.aarch64.neon.sqshl.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
        ret <8 x i16> %tmp3
}

define <4 x i32> @sqshl4s(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: sqshl4s:
;CHECK: sqshl.4s
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp2 = load <4 x i32>, <4 x i32>* %B
        %tmp3 = call <4 x i32> @llvm.aarch64.neon.sqshl.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
        ret <4 x i32> %tmp3
}

define <2 x i64> @sqshl2d(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK-LABEL: sqshl2d:
;CHECK: sqshl.2d
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp2 = load <2 x i64>, <2 x i64>* %B
        %tmp3 = call <2 x i64> @llvm.aarch64.neon.sqshl.v2i64(<2 x i64> %tmp1, <2 x i64> %tmp2)
        ret <2 x i64> %tmp3
}

define <16 x i8> @uqshl16b(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: uqshl16b:
;CHECK: uqshl.16b
        %tmp1 = load <16 x i8>, <16 x i8>* %A
        %tmp2 = load <16 x i8>, <16 x i8>* %B
        %tmp3 = call <16 x i8> @llvm.aarch64.neon.uqshl.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
        ret <16 x i8> %tmp3
}

define <8 x i16> @uqshl8h(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: uqshl8h:
;CHECK: uqshl.8h
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp2 = load <8 x i16>, <8 x i16>* %B
        %tmp3 = call <8 x i16> @llvm.aarch64.neon.uqshl.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
        ret <8 x i16> %tmp3
}

define <4 x i32> @uqshl4s(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: uqshl4s:
;CHECK: uqshl.4s
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp2 = load <4 x i32>, <4 x i32>* %B
        %tmp3 = call <4 x i32> @llvm.aarch64.neon.uqshl.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
        ret <4 x i32> %tmp3
}

define <2 x i64> @uqshl2d(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK-LABEL: uqshl2d:
;CHECK: uqshl.2d
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp2 = load <2 x i64>, <2 x i64>* %B
        %tmp3 = call <2 x i64> @llvm.aarch64.neon.uqshl.v2i64(<2 x i64> %tmp1, <2 x i64> %tmp2)
        ret <2 x i64> %tmp3
}

declare <8 x i8>  @llvm.aarch64.neon.sqshl.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.sqshl.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.aarch64.neon.sqshl.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <1 x i64> @llvm.aarch64.neon.sqshl.v1i64(<1 x i64>, <1 x i64>) nounwind readnone

declare <8 x i8>  @llvm.aarch64.neon.uqshl.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.uqshl.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.aarch64.neon.uqshl.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <1 x i64> @llvm.aarch64.neon.uqshl.v1i64(<1 x i64>, <1 x i64>) nounwind readnone

declare <16 x i8> @llvm.aarch64.neon.sqshl.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.aarch64.neon.sqshl.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.aarch64.neon.sqshl.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i64> @llvm.aarch64.neon.sqshl.v2i64(<2 x i64>, <2 x i64>) nounwind readnone

declare <16 x i8> @llvm.aarch64.neon.uqshl.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.aarch64.neon.uqshl.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.aarch64.neon.uqshl.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i64> @llvm.aarch64.neon.uqshl.v2i64(<2 x i64>, <2 x i64>) nounwind readnone

define <8 x i8> @srshl8b(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: srshl8b:
;CHECK: srshl.8b
        %tmp1 = load <8 x i8>, <8 x i8>* %A
        %tmp2 = load <8 x i8>, <8 x i8>* %B
        %tmp3 = call <8 x i8> @llvm.aarch64.neon.srshl.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
        ret <8 x i8> %tmp3
}

define <4 x i16> @srshl4h(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: srshl4h:
;CHECK: srshl.4h
        %tmp1 = load <4 x i16>, <4 x i16>* %A
        %tmp2 = load <4 x i16>, <4 x i16>* %B
        %tmp3 = call <4 x i16> @llvm.aarch64.neon.srshl.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
        ret <4 x i16> %tmp3
}

define <2 x i32> @srshl2s(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: srshl2s:
;CHECK: srshl.2s
        %tmp1 = load <2 x i32>, <2 x i32>* %A
        %tmp2 = load <2 x i32>, <2 x i32>* %B
        %tmp3 = call <2 x i32> @llvm.aarch64.neon.srshl.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
        ret <2 x i32> %tmp3
}

define <8 x i8> @urshl8b(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: urshl8b:
;CHECK: urshl.8b
        %tmp1 = load <8 x i8>, <8 x i8>* %A
        %tmp2 = load <8 x i8>, <8 x i8>* %B
        %tmp3 = call <8 x i8> @llvm.aarch64.neon.urshl.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
        ret <8 x i8> %tmp3
}

define <4 x i16> @urshl4h(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: urshl4h:
;CHECK: urshl.4h
        %tmp1 = load <4 x i16>, <4 x i16>* %A
        %tmp2 = load <4 x i16>, <4 x i16>* %B
        %tmp3 = call <4 x i16> @llvm.aarch64.neon.urshl.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
        ret <4 x i16> %tmp3
}

define <2 x i32> @urshl2s(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: urshl2s:
;CHECK: urshl.2s
        %tmp1 = load <2 x i32>, <2 x i32>* %A
        %tmp2 = load <2 x i32>, <2 x i32>* %B
        %tmp3 = call <2 x i32> @llvm.aarch64.neon.urshl.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
        ret <2 x i32> %tmp3
}

define <16 x i8> @srshl16b(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: srshl16b:
;CHECK: srshl.16b
        %tmp1 = load <16 x i8>, <16 x i8>* %A
        %tmp2 = load <16 x i8>, <16 x i8>* %B
        %tmp3 = call <16 x i8> @llvm.aarch64.neon.srshl.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
        ret <16 x i8> %tmp3
}

define <8 x i16> @srshl8h(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: srshl8h:
;CHECK: srshl.8h
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp2 = load <8 x i16>, <8 x i16>* %B
        %tmp3 = call <8 x i16> @llvm.aarch64.neon.srshl.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
        ret <8 x i16> %tmp3
}

define <4 x i32> @srshl4s(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: srshl4s:
;CHECK: srshl.4s
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp2 = load <4 x i32>, <4 x i32>* %B
        %tmp3 = call <4 x i32> @llvm.aarch64.neon.srshl.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
        ret <4 x i32> %tmp3
}

define <2 x i64> @srshl2d(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK-LABEL: srshl2d:
;CHECK: srshl.2d
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp2 = load <2 x i64>, <2 x i64>* %B
        %tmp3 = call <2 x i64> @llvm.aarch64.neon.srshl.v2i64(<2 x i64> %tmp1, <2 x i64> %tmp2)
        ret <2 x i64> %tmp3
}

define <16 x i8> @urshl16b(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: urshl16b:
;CHECK: urshl.16b
        %tmp1 = load <16 x i8>, <16 x i8>* %A
        %tmp2 = load <16 x i8>, <16 x i8>* %B
        %tmp3 = call <16 x i8> @llvm.aarch64.neon.urshl.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
        ret <16 x i8> %tmp3
}

define <8 x i16> @urshl8h(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: urshl8h:
;CHECK: urshl.8h
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp2 = load <8 x i16>, <8 x i16>* %B
        %tmp3 = call <8 x i16> @llvm.aarch64.neon.urshl.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
        ret <8 x i16> %tmp3
}

define <4 x i32> @urshl4s(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: urshl4s:
;CHECK: urshl.4s
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp2 = load <4 x i32>, <4 x i32>* %B
        %tmp3 = call <4 x i32> @llvm.aarch64.neon.urshl.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
        ret <4 x i32> %tmp3
}

define <2 x i64> @urshl2d(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK-LABEL: urshl2d:
;CHECK: urshl.2d
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp2 = load <2 x i64>, <2 x i64>* %B
        %tmp3 = call <2 x i64> @llvm.aarch64.neon.urshl.v2i64(<2 x i64> %tmp1, <2 x i64> %tmp2)
        ret <2 x i64> %tmp3
}

declare <8 x i8>  @llvm.aarch64.neon.srshl.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.srshl.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.aarch64.neon.srshl.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <1 x i64> @llvm.aarch64.neon.srshl.v1i64(<1 x i64>, <1 x i64>) nounwind readnone

declare <8 x i8>  @llvm.aarch64.neon.urshl.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.urshl.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.aarch64.neon.urshl.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <1 x i64> @llvm.aarch64.neon.urshl.v1i64(<1 x i64>, <1 x i64>) nounwind readnone

declare <16 x i8> @llvm.aarch64.neon.srshl.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.aarch64.neon.srshl.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.aarch64.neon.srshl.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i64> @llvm.aarch64.neon.srshl.v2i64(<2 x i64>, <2 x i64>) nounwind readnone

declare <16 x i8> @llvm.aarch64.neon.urshl.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.aarch64.neon.urshl.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.aarch64.neon.urshl.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i64> @llvm.aarch64.neon.urshl.v2i64(<2 x i64>, <2 x i64>) nounwind readnone

define <8 x i8> @sqrshl8b(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: sqrshl8b:
;CHECK: sqrshl.8b
        %tmp1 = load <8 x i8>, <8 x i8>* %A
        %tmp2 = load <8 x i8>, <8 x i8>* %B
        %tmp3 = call <8 x i8> @llvm.aarch64.neon.sqrshl.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
        ret <8 x i8> %tmp3
}

define <4 x i16> @sqrshl4h(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: sqrshl4h:
;CHECK: sqrshl.4h
        %tmp1 = load <4 x i16>, <4 x i16>* %A
        %tmp2 = load <4 x i16>, <4 x i16>* %B
        %tmp3 = call <4 x i16> @llvm.aarch64.neon.sqrshl.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
        ret <4 x i16> %tmp3
}

define <2 x i32> @sqrshl2s(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: sqrshl2s:
;CHECK: sqrshl.2s
        %tmp1 = load <2 x i32>, <2 x i32>* %A
        %tmp2 = load <2 x i32>, <2 x i32>* %B
        %tmp3 = call <2 x i32> @llvm.aarch64.neon.sqrshl.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
        ret <2 x i32> %tmp3
}

define <8 x i8> @uqrshl8b(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: uqrshl8b:
;CHECK: uqrshl.8b
        %tmp1 = load <8 x i8>, <8 x i8>* %A
        %tmp2 = load <8 x i8>, <8 x i8>* %B
        %tmp3 = call <8 x i8> @llvm.aarch64.neon.uqrshl.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
        ret <8 x i8> %tmp3
}

define <4 x i16> @uqrshl4h(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: uqrshl4h:
;CHECK: uqrshl.4h
        %tmp1 = load <4 x i16>, <4 x i16>* %A
        %tmp2 = load <4 x i16>, <4 x i16>* %B
        %tmp3 = call <4 x i16> @llvm.aarch64.neon.uqrshl.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
        ret <4 x i16> %tmp3
}

define <2 x i32> @uqrshl2s(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: uqrshl2s:
;CHECK: uqrshl.2s
        %tmp1 = load <2 x i32>, <2 x i32>* %A
        %tmp2 = load <2 x i32>, <2 x i32>* %B
        %tmp3 = call <2 x i32> @llvm.aarch64.neon.uqrshl.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
        ret <2 x i32> %tmp3
}

define <16 x i8> @sqrshl16b(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: sqrshl16b:
;CHECK: sqrshl.16b
        %tmp1 = load <16 x i8>, <16 x i8>* %A
        %tmp2 = load <16 x i8>, <16 x i8>* %B
        %tmp3 = call <16 x i8> @llvm.aarch64.neon.sqrshl.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
        ret <16 x i8> %tmp3
}

define <8 x i16> @sqrshl8h(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: sqrshl8h:
;CHECK: sqrshl.8h
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp2 = load <8 x i16>, <8 x i16>* %B
        %tmp3 = call <8 x i16> @llvm.aarch64.neon.sqrshl.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
        ret <8 x i16> %tmp3
}

define <4 x i32> @sqrshl4s(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: sqrshl4s:
;CHECK: sqrshl.4s
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp2 = load <4 x i32>, <4 x i32>* %B
        %tmp3 = call <4 x i32> @llvm.aarch64.neon.sqrshl.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
        ret <4 x i32> %tmp3
}

define <2 x i64> @sqrshl2d(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK-LABEL: sqrshl2d:
;CHECK: sqrshl.2d
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp2 = load <2 x i64>, <2 x i64>* %B
        %tmp3 = call <2 x i64> @llvm.aarch64.neon.sqrshl.v2i64(<2 x i64> %tmp1, <2 x i64> %tmp2)
        ret <2 x i64> %tmp3
}

define <16 x i8> @uqrshl16b(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: uqrshl16b:
;CHECK: uqrshl.16b
        %tmp1 = load <16 x i8>, <16 x i8>* %A
        %tmp2 = load <16 x i8>, <16 x i8>* %B
        %tmp3 = call <16 x i8> @llvm.aarch64.neon.uqrshl.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
        ret <16 x i8> %tmp3
}

define <8 x i16> @uqrshl8h(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: uqrshl8h:
;CHECK: uqrshl.8h
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp2 = load <8 x i16>, <8 x i16>* %B
        %tmp3 = call <8 x i16> @llvm.aarch64.neon.uqrshl.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
        ret <8 x i16> %tmp3
}

define <4 x i32> @uqrshl4s(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: uqrshl4s:
;CHECK: uqrshl.4s
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp2 = load <4 x i32>, <4 x i32>* %B
        %tmp3 = call <4 x i32> @llvm.aarch64.neon.uqrshl.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
        ret <4 x i32> %tmp3
}

define <2 x i64> @uqrshl2d(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK-LABEL: uqrshl2d:
;CHECK: uqrshl.2d
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp2 = load <2 x i64>, <2 x i64>* %B
        %tmp3 = call <2 x i64> @llvm.aarch64.neon.uqrshl.v2i64(<2 x i64> %tmp1, <2 x i64> %tmp2)
        ret <2 x i64> %tmp3
}

declare <8 x i8>  @llvm.aarch64.neon.sqrshl.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.sqrshl.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.aarch64.neon.sqrshl.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <1 x i64> @llvm.aarch64.neon.sqrshl.v1i64(<1 x i64>, <1 x i64>) nounwind readnone

declare <8 x i8>  @llvm.aarch64.neon.uqrshl.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.uqrshl.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.aarch64.neon.uqrshl.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <1 x i64> @llvm.aarch64.neon.uqrshl.v1i64(<1 x i64>, <1 x i64>) nounwind readnone

declare <16 x i8> @llvm.aarch64.neon.sqrshl.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.aarch64.neon.sqrshl.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.aarch64.neon.sqrshl.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i64> @llvm.aarch64.neon.sqrshl.v2i64(<2 x i64>, <2 x i64>) nounwind readnone

declare <16 x i8> @llvm.aarch64.neon.uqrshl.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.aarch64.neon.uqrshl.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.aarch64.neon.uqrshl.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i64> @llvm.aarch64.neon.uqrshl.v2i64(<2 x i64>, <2 x i64>) nounwind readnone

define <8 x i8> @urshr8b(<8 x i8>* %A) nounwind {
;CHECK-LABEL: urshr8b:
;CHECK: urshr.8b
        %tmp1 = load <8 x i8>, <8 x i8>* %A
        %tmp3 = call <8 x i8> @llvm.aarch64.neon.urshl.v8i8(<8 x i8> %tmp1, <8 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>)
        ret <8 x i8> %tmp3
}

define <4 x i16> @urshr4h(<4 x i16>* %A) nounwind {
;CHECK-LABEL: urshr4h:
;CHECK: urshr.4h
        %tmp1 = load <4 x i16>, <4 x i16>* %A
        %tmp3 = call <4 x i16> @llvm.aarch64.neon.urshl.v4i16(<4 x i16> %tmp1, <4 x i16> <i16 -1, i16 -1, i16 -1, i16 -1>)
        ret <4 x i16> %tmp3
}

define <2 x i32> @urshr2s(<2 x i32>* %A) nounwind {
;CHECK-LABEL: urshr2s:
;CHECK: urshr.2s
        %tmp1 = load <2 x i32>, <2 x i32>* %A
        %tmp3 = call <2 x i32> @llvm.aarch64.neon.urshl.v2i32(<2 x i32> %tmp1, <2 x i32> <i32 -1, i32 -1>)
        ret <2 x i32> %tmp3
}

define <16 x i8> @urshr16b(<16 x i8>* %A) nounwind {
;CHECK-LABEL: urshr16b:
;CHECK: urshr.16b
        %tmp1 = load <16 x i8>, <16 x i8>* %A
        %tmp3 = call <16 x i8> @llvm.aarch64.neon.urshl.v16i8(<16 x i8> %tmp1, <16 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>)
        ret <16 x i8> %tmp3
}

define <8 x i16> @urshr8h(<8 x i16>* %A) nounwind {
;CHECK-LABEL: urshr8h:
;CHECK: urshr.8h
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp3 = call <8 x i16> @llvm.aarch64.neon.urshl.v8i16(<8 x i16> %tmp1, <8 x i16> <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>)
        ret <8 x i16> %tmp3
}

define <4 x i32> @urshr4s(<4 x i32>* %A) nounwind {
;CHECK-LABEL: urshr4s:
;CHECK: urshr.4s
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp3 = call <4 x i32> @llvm.aarch64.neon.urshl.v4i32(<4 x i32> %tmp1, <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>)
        ret <4 x i32> %tmp3
}

define <2 x i64> @urshr2d(<2 x i64>* %A) nounwind {
;CHECK-LABEL: urshr2d:
;CHECK: urshr.2d
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp3 = call <2 x i64> @llvm.aarch64.neon.urshl.v2i64(<2 x i64> %tmp1, <2 x i64> <i64 -1, i64 -1>)
        ret <2 x i64> %tmp3
}

define <8 x i8> @srshr8b(<8 x i8>* %A) nounwind {
;CHECK-LABEL: srshr8b:
;CHECK: srshr.8b
        %tmp1 = load <8 x i8>, <8 x i8>* %A
        %tmp3 = call <8 x i8> @llvm.aarch64.neon.srshl.v8i8(<8 x i8> %tmp1, <8 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>)
        ret <8 x i8> %tmp3
}

define <4 x i16> @srshr4h(<4 x i16>* %A) nounwind {
;CHECK-LABEL: srshr4h:
;CHECK: srshr.4h
        %tmp1 = load <4 x i16>, <4 x i16>* %A
        %tmp3 = call <4 x i16> @llvm.aarch64.neon.srshl.v4i16(<4 x i16> %tmp1, <4 x i16> <i16 -1, i16 -1, i16 -1, i16 -1>)
        ret <4 x i16> %tmp3
}

define <2 x i32> @srshr2s(<2 x i32>* %A) nounwind {
;CHECK-LABEL: srshr2s:
;CHECK: srshr.2s
        %tmp1 = load <2 x i32>, <2 x i32>* %A
        %tmp3 = call <2 x i32> @llvm.aarch64.neon.srshl.v2i32(<2 x i32> %tmp1, <2 x i32> <i32 -1, i32 -1>)
        ret <2 x i32> %tmp3
}

define <16 x i8> @srshr16b(<16 x i8>* %A) nounwind {
;CHECK-LABEL: srshr16b:
;CHECK: srshr.16b
        %tmp1 = load <16 x i8>, <16 x i8>* %A
        %tmp3 = call <16 x i8> @llvm.aarch64.neon.srshl.v16i8(<16 x i8> %tmp1, <16 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>)
        ret <16 x i8> %tmp3
}

define <8 x i16> @srshr8h(<8 x i16>* %A) nounwind {
;CHECK-LABEL: srshr8h:
;CHECK: srshr.8h
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp3 = call <8 x i16> @llvm.aarch64.neon.srshl.v8i16(<8 x i16> %tmp1, <8 x i16> <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>)
        ret <8 x i16> %tmp3
}

define <4 x i32> @srshr4s(<4 x i32>* %A) nounwind {
;CHECK-LABEL: srshr4s:
;CHECK: srshr.4s
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp3 = call <4 x i32> @llvm.aarch64.neon.srshl.v4i32(<4 x i32> %tmp1, <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>)
        ret <4 x i32> %tmp3
}

define <2 x i64> @srshr2d(<2 x i64>* %A) nounwind {
;CHECK-LABEL: srshr2d:
;CHECK: srshr.2d
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp3 = call <2 x i64> @llvm.aarch64.neon.srshl.v2i64(<2 x i64> %tmp1, <2 x i64> <i64 -1, i64 -1>)
        ret <2 x i64> %tmp3
}

define <8 x i8> @sqshlu8b(<8 x i8>* %A) nounwind {
;CHECK-LABEL: sqshlu8b:
;CHECK: sqshlu.8b v0, {{v[0-9]+}}, #1
        %tmp1 = load <8 x i8>, <8 x i8>* %A
        %tmp3 = call <8 x i8> @llvm.aarch64.neon.sqshlu.v8i8(<8 x i8> %tmp1, <8 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>)
        ret <8 x i8> %tmp3
}

define <4 x i16> @sqshlu4h(<4 x i16>* %A) nounwind {
;CHECK-LABEL: sqshlu4h:
;CHECK: sqshlu.4h v0, {{v[0-9]+}}, #1
        %tmp1 = load <4 x i16>, <4 x i16>* %A
        %tmp3 = call <4 x i16> @llvm.aarch64.neon.sqshlu.v4i16(<4 x i16> %tmp1, <4 x i16> <i16 1, i16 1, i16 1, i16 1>)
        ret <4 x i16> %tmp3
}

define <2 x i32> @sqshlu2s(<2 x i32>* %A) nounwind {
;CHECK-LABEL: sqshlu2s:
;CHECK: sqshlu.2s v0, {{v[0-9]+}}, #1
        %tmp1 = load <2 x i32>, <2 x i32>* %A
        %tmp3 = call <2 x i32> @llvm.aarch64.neon.sqshlu.v2i32(<2 x i32> %tmp1, <2 x i32> <i32 1, i32 1>)
        ret <2 x i32> %tmp3
}

define <16 x i8> @sqshlu16b(<16 x i8>* %A) nounwind {
;CHECK-LABEL: sqshlu16b:
;CHECK: sqshlu.16b v0, {{v[0-9]+}}, #1
        %tmp1 = load <16 x i8>, <16 x i8>* %A
        %tmp3 = call <16 x i8> @llvm.aarch64.neon.sqshlu.v16i8(<16 x i8> %tmp1, <16 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>)
        ret <16 x i8> %tmp3
}

define <8 x i16> @sqshlu8h(<8 x i16>* %A) nounwind {
;CHECK-LABEL: sqshlu8h:
;CHECK: sqshlu.8h v0, {{v[0-9]+}}, #1
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp3 = call <8 x i16> @llvm.aarch64.neon.sqshlu.v8i16(<8 x i16> %tmp1, <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>)
        ret <8 x i16> %tmp3
}

define <4 x i32> @sqshlu4s(<4 x i32>* %A) nounwind {
;CHECK-LABEL: sqshlu4s:
;CHECK: sqshlu.4s v0, {{v[0-9]+}}, #1
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp3 = call <4 x i32> @llvm.aarch64.neon.sqshlu.v4i32(<4 x i32> %tmp1, <4 x i32> <i32 1, i32 1, i32 1, i32 1>)
        ret <4 x i32> %tmp3
}

define <2 x i64> @sqshlu2d(<2 x i64>* %A) nounwind {
;CHECK-LABEL: sqshlu2d:
;CHECK: sqshlu.2d v0, {{v[0-9]+}}, #1
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp3 = call <2 x i64> @llvm.aarch64.neon.sqshlu.v2i64(<2 x i64> %tmp1, <2 x i64> <i64 1, i64 1>)
        ret <2 x i64> %tmp3
}

declare <8 x i8>  @llvm.aarch64.neon.sqshlu.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.sqshlu.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.aarch64.neon.sqshlu.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <1 x i64> @llvm.aarch64.neon.sqshlu.v1i64(<1 x i64>, <1 x i64>) nounwind readnone

declare <16 x i8> @llvm.aarch64.neon.sqshlu.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.aarch64.neon.sqshlu.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.aarch64.neon.sqshlu.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i64> @llvm.aarch64.neon.sqshlu.v2i64(<2 x i64>, <2 x i64>) nounwind readnone

define <8 x i8> @rshrn8b(<8 x i16>* %A) nounwind {
;CHECK-LABEL: rshrn8b:
;CHECK: rshrn.8b v0, {{v[0-9]+}}, #1
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp3 = call <8 x i8> @llvm.aarch64.neon.rshrn.v8i8(<8 x i16> %tmp1, i32 1)
        ret <8 x i8> %tmp3
}

define <4 x i16> @rshrn4h(<4 x i32>* %A) nounwind {
;CHECK-LABEL: rshrn4h:
;CHECK: rshrn.4h v0, {{v[0-9]+}}, #1
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp3 = call <4 x i16> @llvm.aarch64.neon.rshrn.v4i16(<4 x i32> %tmp1, i32 1)
        ret <4 x i16> %tmp3
}

define <2 x i32> @rshrn2s(<2 x i64>* %A) nounwind {
;CHECK-LABEL: rshrn2s:
;CHECK: rshrn.2s v0, {{v[0-9]+}}, #1
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp3 = call <2 x i32> @llvm.aarch64.neon.rshrn.v2i32(<2 x i64> %tmp1, i32 1)
        ret <2 x i32> %tmp3
}

define <16 x i8> @rshrn16b(<8 x i8> *%ret, <8 x i16>* %A) nounwind {
;CHECK-LABEL: rshrn16b:
;CHECK: rshrn2.16b v0, {{v[0-9]+}}, #1
        %out = load <8 x i8>, <8 x i8>* %ret
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp3 = call <8 x i8> @llvm.aarch64.neon.rshrn.v8i8(<8 x i16> %tmp1, i32 1)
        %tmp4 = shufflevector <8 x i8> %out, <8 x i8> %tmp3, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
        ret <16 x i8> %tmp4
}

define <8 x i16> @rshrn8h(<4 x i16>* %ret, <4 x i32>* %A) nounwind {
;CHECK-LABEL: rshrn8h:
;CHECK: rshrn2.8h v0, {{v[0-9]+}}, #1
        %out = load <4 x i16>, <4 x i16>* %ret
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp3 = call <4 x i16> @llvm.aarch64.neon.rshrn.v4i16(<4 x i32> %tmp1, i32 1)
        %tmp4 = shufflevector <4 x i16> %out, <4 x i16> %tmp3, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
        ret <8 x i16> %tmp4
}

define <4 x i32> @rshrn4s(<2 x i32>* %ret, <2 x i64>* %A) nounwind {
;CHECK-LABEL: rshrn4s:
;CHECK: rshrn2.4s v0, {{v[0-9]+}}, #1
        %out = load <2 x i32>, <2 x i32>* %ret
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp3 = call <2 x i32> @llvm.aarch64.neon.rshrn.v2i32(<2 x i64> %tmp1, i32 1)
        %tmp4 = shufflevector <2 x i32> %out, <2 x i32> %tmp3, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
        ret <4 x i32> %tmp4
}

declare <8 x i8>  @llvm.aarch64.neon.rshrn.v8i8(<8 x i16>, i32) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.rshrn.v4i16(<4 x i32>, i32) nounwind readnone
declare <2 x i32> @llvm.aarch64.neon.rshrn.v2i32(<2 x i64>, i32) nounwind readnone

define <8 x i8> @shrn8b(<8 x i16>* %A) nounwind {
;CHECK-LABEL: shrn8b:
;CHECK: shrn.8b v0, {{v[0-9]+}}, #1
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp2 = lshr <8 x i16> %tmp1, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
        %tmp3 = trunc <8 x i16> %tmp2 to <8 x i8>
        ret <8 x i8> %tmp3
}

define <4 x i16> @shrn4h(<4 x i32>* %A) nounwind {
;CHECK-LABEL: shrn4h:
;CHECK: shrn.4h v0, {{v[0-9]+}}, #1
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp2 = lshr <4 x i32> %tmp1, <i32 1, i32 1, i32 1, i32 1>
        %tmp3 = trunc <4 x i32> %tmp2 to <4 x i16>
        ret <4 x i16> %tmp3
}

define <2 x i32> @shrn2s(<2 x i64>* %A) nounwind {
;CHECK-LABEL: shrn2s:
;CHECK: shrn.2s v0, {{v[0-9]+}}, #1
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp2 = lshr <2 x i64> %tmp1, <i64 1, i64 1>
        %tmp3 = trunc <2 x i64> %tmp2 to <2 x i32>
        ret <2 x i32> %tmp3
}

define <16 x i8> @shrn16b(<8 x i8>* %ret, <8 x i16>* %A) nounwind {
;CHECK-LABEL: shrn16b:
;CHECK: shrn2.16b v0, {{v[0-9]+}}, #1
        %out = load <8 x i8>, <8 x i8>* %ret
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp2 = lshr <8 x i16> %tmp1, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
        %tmp3 = trunc <8 x i16> %tmp2 to <8 x i8>
        %tmp4 = shufflevector <8 x i8> %out, <8 x i8> %tmp3, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
        ret <16 x i8> %tmp4
}

define <8 x i16> @shrn8h(<4 x i16>* %ret, <4 x i32>* %A) nounwind {
;CHECK-LABEL: shrn8h:
;CHECK: shrn2.8h v0, {{v[0-9]+}}, #1
        %out = load <4 x i16>, <4 x i16>* %ret
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp2 = lshr <4 x i32> %tmp1, <i32 1, i32 1, i32 1, i32 1>
        %tmp3 = trunc <4 x i32> %tmp2 to <4 x i16>
        %tmp4 = shufflevector <4 x i16> %out, <4 x i16> %tmp3, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
        ret <8 x i16> %tmp4
}

define <4 x i32> @shrn4s(<2 x i32>* %ret, <2 x i64>* %A) nounwind {
;CHECK-LABEL: shrn4s:
;CHECK: shrn2.4s v0, {{v[0-9]+}}, #1
        %out = load <2 x i32>, <2 x i32>* %ret
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp2 = lshr <2 x i64> %tmp1, <i64 1, i64 1>
        %tmp3 = trunc <2 x i64> %tmp2 to <2 x i32>
        %tmp4 = shufflevector <2 x i32> %out, <2 x i32> %tmp3, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
        ret <4 x i32> %tmp4
}

declare <8 x i8>  @llvm.aarch64.neon.shrn.v8i8(<8 x i16>, i32) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.shrn.v4i16(<4 x i32>, i32) nounwind readnone
declare <2 x i32> @llvm.aarch64.neon.shrn.v2i32(<2 x i64>, i32) nounwind readnone

define i32 @sqshrn1s(i64 %A) nounwind {
; CHECK-LABEL: sqshrn1s:
; CHECK: sqshrn {{s[0-9]+}}, d0, #1
  %tmp = call i32 @llvm.aarch64.neon.sqshrn.i32(i64 %A, i32 1)
  ret i32 %tmp
}

define <8 x i8> @sqshrn8b(<8 x i16>* %A) nounwind {
;CHECK-LABEL: sqshrn8b:
;CHECK: sqshrn.8b v0, {{v[0-9]+}}, #1
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp3 = call <8 x i8> @llvm.aarch64.neon.sqshrn.v8i8(<8 x i16> %tmp1, i32 1)
        ret <8 x i8> %tmp3
}

define <4 x i16> @sqshrn4h(<4 x i32>* %A) nounwind {
;CHECK-LABEL: sqshrn4h:
;CHECK: sqshrn.4h v0, {{v[0-9]+}}, #1
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp3 = call <4 x i16> @llvm.aarch64.neon.sqshrn.v4i16(<4 x i32> %tmp1, i32 1)
        ret <4 x i16> %tmp3
}

define <2 x i32> @sqshrn2s(<2 x i64>* %A) nounwind {
;CHECK-LABEL: sqshrn2s:
;CHECK: sqshrn.2s v0, {{v[0-9]+}}, #1
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp3 = call <2 x i32> @llvm.aarch64.neon.sqshrn.v2i32(<2 x i64> %tmp1, i32 1)
        ret <2 x i32> %tmp3
}


define <16 x i8> @sqshrn16b(<8 x i8>* %ret, <8 x i16>* %A) nounwind {
;CHECK-LABEL: sqshrn16b:
;CHECK: sqshrn2.16b v0, {{v[0-9]+}}, #1
        %out = load <8 x i8>, <8 x i8>* %ret
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp3 = call <8 x i8> @llvm.aarch64.neon.sqshrn.v8i8(<8 x i16> %tmp1, i32 1)
        %tmp4 = shufflevector <8 x i8> %out, <8 x i8> %tmp3, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
        ret <16 x i8> %tmp4
}

define <8 x i16> @sqshrn8h(<4 x i16>* %ret, <4 x i32>* %A) nounwind {
;CHECK-LABEL: sqshrn8h:
;CHECK: sqshrn2.8h v0, {{v[0-9]+}}, #1
        %out = load <4 x i16>, <4 x i16>* %ret
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp3 = call <4 x i16> @llvm.aarch64.neon.sqshrn.v4i16(<4 x i32> %tmp1, i32 1)
        %tmp4 = shufflevector <4 x i16> %out, <4 x i16> %tmp3, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
        ret <8 x i16> %tmp4
}

define <4 x i32> @sqshrn4s(<2 x i32>* %ret, <2 x i64>* %A) nounwind {
;CHECK-LABEL: sqshrn4s:
;CHECK: sqshrn2.4s v0, {{v[0-9]+}}, #1
        %out = load <2 x i32>, <2 x i32>* %ret
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp3 = call <2 x i32> @llvm.aarch64.neon.sqshrn.v2i32(<2 x i64> %tmp1, i32 1)
        %tmp4 = shufflevector <2 x i32> %out, <2 x i32> %tmp3, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
        ret <4 x i32> %tmp4
}

declare i32  @llvm.aarch64.neon.sqshrn.i32(i64, i32) nounwind readnone
declare <8 x i8>  @llvm.aarch64.neon.sqshrn.v8i8(<8 x i16>, i32) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.sqshrn.v4i16(<4 x i32>, i32) nounwind readnone
declare <2 x i32> @llvm.aarch64.neon.sqshrn.v2i32(<2 x i64>, i32) nounwind readnone

define i32 @sqshrun1s(i64 %A) nounwind {
; CHECK-LABEL: sqshrun1s:
; CHECK: sqshrun {{s[0-9]+}}, d0, #1
  %tmp = call i32 @llvm.aarch64.neon.sqshrun.i32(i64 %A, i32 1)
  ret i32 %tmp
}

define <8 x i8> @sqshrun8b(<8 x i16>* %A) nounwind {
;CHECK-LABEL: sqshrun8b:
;CHECK: sqshrun.8b v0, {{v[0-9]+}}, #1
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp3 = call <8 x i8> @llvm.aarch64.neon.sqshrun.v8i8(<8 x i16> %tmp1, i32 1)
        ret <8 x i8> %tmp3
}

define <4 x i16> @sqshrun4h(<4 x i32>* %A) nounwind {
;CHECK-LABEL: sqshrun4h:
;CHECK: sqshrun.4h v0, {{v[0-9]+}}, #1
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp3 = call <4 x i16> @llvm.aarch64.neon.sqshrun.v4i16(<4 x i32> %tmp1, i32 1)
        ret <4 x i16> %tmp3
}

define <2 x i32> @sqshrun2s(<2 x i64>* %A) nounwind {
;CHECK-LABEL: sqshrun2s:
;CHECK: sqshrun.2s v0, {{v[0-9]+}}, #1
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp3 = call <2 x i32> @llvm.aarch64.neon.sqshrun.v2i32(<2 x i64> %tmp1, i32 1)
        ret <2 x i32> %tmp3
}

define <16 x i8> @sqshrun16b(<8 x i8>* %ret, <8 x i16>* %A) nounwind {
;CHECK-LABEL: sqshrun16b:
;CHECK: sqshrun2.16b v0, {{v[0-9]+}}, #1
        %out = load <8 x i8>, <8 x i8>* %ret
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp3 = call <8 x i8> @llvm.aarch64.neon.sqshrun.v8i8(<8 x i16> %tmp1, i32 1)
        %tmp4 = shufflevector <8 x i8> %out, <8 x i8> %tmp3, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
        ret <16 x i8> %tmp4
}

define <8 x i16> @sqshrun8h(<4 x i16>* %ret, <4 x i32>* %A) nounwind {
;CHECK-LABEL: sqshrun8h:
;CHECK: sqshrun2.8h v0, {{v[0-9]+}}, #1
        %out = load <4 x i16>, <4 x i16>* %ret
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp3 = call <4 x i16> @llvm.aarch64.neon.sqshrun.v4i16(<4 x i32> %tmp1, i32 1)
        %tmp4 = shufflevector <4 x i16> %out, <4 x i16> %tmp3, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
        ret <8 x i16> %tmp4
}

define <4 x i32> @sqshrun4s(<2 x i32>* %ret, <2 x i64>* %A) nounwind {
;CHECK-LABEL: sqshrun4s:
;CHECK: sqshrun2.4s v0, {{v[0-9]+}}, #1
        %out = load <2 x i32>, <2 x i32>* %ret
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp3 = call <2 x i32> @llvm.aarch64.neon.sqshrun.v2i32(<2 x i64> %tmp1, i32 1)
        %tmp4 = shufflevector <2 x i32> %out, <2 x i32> %tmp3, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
        ret <4 x i32> %tmp4
}

declare i32  @llvm.aarch64.neon.sqshrun.i32(i64, i32) nounwind readnone
declare <8 x i8>  @llvm.aarch64.neon.sqshrun.v8i8(<8 x i16>, i32) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.sqshrun.v4i16(<4 x i32>, i32) nounwind readnone
declare <2 x i32> @llvm.aarch64.neon.sqshrun.v2i32(<2 x i64>, i32) nounwind readnone

define i32 @sqrshrn1s(i64 %A) nounwind {
; CHECK-LABEL: sqrshrn1s:
; CHECK: sqrshrn {{s[0-9]+}}, d0, #1
  %tmp = call i32 @llvm.aarch64.neon.sqrshrn.i32(i64 %A, i32 1)
  ret i32 %tmp
}

define <8 x i8> @sqrshrn8b(<8 x i16>* %A) nounwind {
;CHECK-LABEL: sqrshrn8b:
;CHECK: sqrshrn.8b v0, {{v[0-9]+}}, #1
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp3 = call <8 x i8> @llvm.aarch64.neon.sqrshrn.v8i8(<8 x i16> %tmp1, i32 1)
        ret <8 x i8> %tmp3
}

define <4 x i16> @sqrshrn4h(<4 x i32>* %A) nounwind {
;CHECK-LABEL: sqrshrn4h:
;CHECK: sqrshrn.4h v0, {{v[0-9]+}}, #1
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp3 = call <4 x i16> @llvm.aarch64.neon.sqrshrn.v4i16(<4 x i32> %tmp1, i32 1)
        ret <4 x i16> %tmp3
}

define <2 x i32> @sqrshrn2s(<2 x i64>* %A) nounwind {
;CHECK-LABEL: sqrshrn2s:
;CHECK: sqrshrn.2s v0, {{v[0-9]+}}, #1
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp3 = call <2 x i32> @llvm.aarch64.neon.sqrshrn.v2i32(<2 x i64> %tmp1, i32 1)
        ret <2 x i32> %tmp3
}

define <16 x i8> @sqrshrn16b(<8 x i8>* %ret, <8 x i16>* %A) nounwind {
;CHECK-LABEL: sqrshrn16b:
;CHECK: sqrshrn2.16b v0, {{v[0-9]+}}, #1
        %out = load <8 x i8>, <8 x i8>* %ret
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp3 = call <8 x i8> @llvm.aarch64.neon.sqrshrn.v8i8(<8 x i16> %tmp1, i32 1)
        %tmp4 = shufflevector <8 x i8> %out, <8 x i8> %tmp3, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
        ret <16 x i8> %tmp4
}

define <8 x i16> @sqrshrn8h(<4 x i16>* %ret, <4 x i32>* %A) nounwind {
;CHECK-LABEL: sqrshrn8h:
;CHECK: sqrshrn2.8h v0, {{v[0-9]+}}, #1
        %out = load <4 x i16>, <4 x i16>* %ret
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp3 = call <4 x i16> @llvm.aarch64.neon.sqrshrn.v4i16(<4 x i32> %tmp1, i32 1)
        %tmp4 = shufflevector <4 x i16> %out, <4 x i16> %tmp3, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
        ret <8 x i16> %tmp4
}

define <4 x i32> @sqrshrn4s(<2 x i32>* %ret, <2 x i64>* %A) nounwind {
;CHECK-LABEL: sqrshrn4s:
;CHECK: sqrshrn2.4s v0, {{v[0-9]+}}, #1
        %out = load <2 x i32>, <2 x i32>* %ret
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp3 = call <2 x i32> @llvm.aarch64.neon.sqrshrn.v2i32(<2 x i64> %tmp1, i32 1)
        %tmp4 = shufflevector <2 x i32> %out, <2 x i32> %tmp3, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
        ret <4 x i32> %tmp4
}

declare i32  @llvm.aarch64.neon.sqrshrn.i32(i64, i32) nounwind readnone
declare <8 x i8>  @llvm.aarch64.neon.sqrshrn.v8i8(<8 x i16>, i32) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.sqrshrn.v4i16(<4 x i32>, i32) nounwind readnone
declare <2 x i32> @llvm.aarch64.neon.sqrshrn.v2i32(<2 x i64>, i32) nounwind readnone

define i32 @sqrshrun1s(i64 %A) nounwind {
; CHECK-LABEL: sqrshrun1s:
; CHECK: sqrshrun {{s[0-9]+}}, d0, #1
  %tmp = call i32 @llvm.aarch64.neon.sqrshrun.i32(i64 %A, i32 1)
  ret i32 %tmp
}

define <8 x i8> @sqrshrun8b(<8 x i16>* %A) nounwind {
;CHECK-LABEL: sqrshrun8b:
;CHECK: sqrshrun.8b v0, {{v[0-9]+}}, #1
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp3 = call <8 x i8> @llvm.aarch64.neon.sqrshrun.v8i8(<8 x i16> %tmp1, i32 1)
        ret <8 x i8> %tmp3
}

define <4 x i16> @sqrshrun4h(<4 x i32>* %A) nounwind {
;CHECK-LABEL: sqrshrun4h:
;CHECK: sqrshrun.4h v0, {{v[0-9]+}}, #1
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp3 = call <4 x i16> @llvm.aarch64.neon.sqrshrun.v4i16(<4 x i32> %tmp1, i32 1)
        ret <4 x i16> %tmp3
}

define <2 x i32> @sqrshrun2s(<2 x i64>* %A) nounwind {
;CHECK-LABEL: sqrshrun2s:
;CHECK: sqrshrun.2s v0, {{v[0-9]+}}, #1
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp3 = call <2 x i32> @llvm.aarch64.neon.sqrshrun.v2i32(<2 x i64> %tmp1, i32 1)
        ret <2 x i32> %tmp3
}

define <16 x i8> @sqrshrun16b(<8 x i8>* %ret, <8 x i16>* %A) nounwind {
;CHECK-LABEL: sqrshrun16b:
;CHECK: sqrshrun2.16b v0, {{v[0-9]+}}, #1
        %out = load <8 x i8>, <8 x i8>* %ret
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp3 = call <8 x i8> @llvm.aarch64.neon.sqrshrun.v8i8(<8 x i16> %tmp1, i32 1)
        %tmp4 = shufflevector <8 x i8> %out, <8 x i8> %tmp3, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
        ret <16 x i8> %tmp4
}

define <8 x i16> @sqrshrun8h(<4 x i16>* %ret, <4 x i32>* %A) nounwind {
;CHECK-LABEL: sqrshrun8h:
;CHECK: sqrshrun2.8h v0, {{v[0-9]+}}, #1
        %out = load <4 x i16>, <4 x i16>* %ret
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp3 = call <4 x i16> @llvm.aarch64.neon.sqrshrun.v4i16(<4 x i32> %tmp1, i32 1)
        %tmp4 = shufflevector <4 x i16> %out, <4 x i16> %tmp3, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
        ret <8 x i16> %tmp4
}

define <4 x i32> @sqrshrun4s(<2 x i32>* %ret, <2 x i64>* %A) nounwind {
;CHECK-LABEL: sqrshrun4s:
;CHECK: sqrshrun2.4s v0, {{v[0-9]+}}, #1
        %out = load <2 x i32>, <2 x i32>* %ret
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp3 = call <2 x i32> @llvm.aarch64.neon.sqrshrun.v2i32(<2 x i64> %tmp1, i32 1)
        %tmp4 = shufflevector <2 x i32> %out, <2 x i32> %tmp3, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
        ret <4 x i32> %tmp4
}

declare i32  @llvm.aarch64.neon.sqrshrun.i32(i64, i32) nounwind readnone
declare <8 x i8>  @llvm.aarch64.neon.sqrshrun.v8i8(<8 x i16>, i32) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.sqrshrun.v4i16(<4 x i32>, i32) nounwind readnone
declare <2 x i32> @llvm.aarch64.neon.sqrshrun.v2i32(<2 x i64>, i32) nounwind readnone

define i32 @uqrshrn1s(i64 %A) nounwind {
; CHECK-LABEL: uqrshrn1s:
; CHECK: uqrshrn {{s[0-9]+}}, d0, #1
  %tmp = call i32 @llvm.aarch64.neon.uqrshrn.i32(i64 %A, i32 1)
  ret i32 %tmp
}

define <8 x i8> @uqrshrn8b(<8 x i16>* %A) nounwind {
;CHECK-LABEL: uqrshrn8b:
;CHECK: uqrshrn.8b v0, {{v[0-9]+}}, #1
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp3 = call <8 x i8> @llvm.aarch64.neon.uqrshrn.v8i8(<8 x i16> %tmp1, i32 1)
        ret <8 x i8> %tmp3
}

define <4 x i16> @uqrshrn4h(<4 x i32>* %A) nounwind {
;CHECK-LABEL: uqrshrn4h:
;CHECK: uqrshrn.4h v0, {{v[0-9]+}}, #1
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp3 = call <4 x i16> @llvm.aarch64.neon.uqrshrn.v4i16(<4 x i32> %tmp1, i32 1)
        ret <4 x i16> %tmp3
}

define <2 x i32> @uqrshrn2s(<2 x i64>* %A) nounwind {
;CHECK-LABEL: uqrshrn2s:
;CHECK: uqrshrn.2s v0, {{v[0-9]+}}, #1
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp3 = call <2 x i32> @llvm.aarch64.neon.uqrshrn.v2i32(<2 x i64> %tmp1, i32 1)
        ret <2 x i32> %tmp3
}

define <16 x i8> @uqrshrn16b(<8 x i8>* %ret, <8 x i16>* %A) nounwind {
;CHECK-LABEL: uqrshrn16b:
;CHECK: uqrshrn2.16b v0, {{v[0-9]+}}, #1
        %out = load <8 x i8>, <8 x i8>* %ret
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp3 = call <8 x i8> @llvm.aarch64.neon.uqrshrn.v8i8(<8 x i16> %tmp1, i32 1)
        %tmp4 = shufflevector <8 x i8> %out, <8 x i8> %tmp3, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
        ret <16 x i8> %tmp4
}

define <8 x i16> @uqrshrn8h(<4 x i16>* %ret, <4 x i32>* %A) nounwind {
;CHECK-LABEL: uqrshrn8h:
;CHECK: uqrshrn2.8h v0, {{v[0-9]+}}, #1
        %out = load <4 x i16>, <4 x i16>* %ret
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp3 = call <4 x i16> @llvm.aarch64.neon.uqrshrn.v4i16(<4 x i32> %tmp1, i32 1)
        %tmp4 = shufflevector <4 x i16> %out, <4 x i16> %tmp3, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
        ret <8 x i16> %tmp4
}

define <4 x i32> @uqrshrn4s(<2 x i32>* %ret, <2 x i64>* %A) nounwind {
;CHECK-LABEL: uqrshrn4s:
;CHECK: uqrshrn2.4s v0, {{v[0-9]+}}, #1
        %out = load <2 x i32>, <2 x i32>* %ret
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp3 = call <2 x i32> @llvm.aarch64.neon.uqrshrn.v2i32(<2 x i64> %tmp1, i32 1)
        %tmp4 = shufflevector <2 x i32> %out, <2 x i32> %tmp3, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
        ret <4 x i32> %tmp4
}

declare i32  @llvm.aarch64.neon.uqrshrn.i32(i64, i32) nounwind readnone
declare <8 x i8>  @llvm.aarch64.neon.uqrshrn.v8i8(<8 x i16>, i32) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.uqrshrn.v4i16(<4 x i32>, i32) nounwind readnone
declare <2 x i32> @llvm.aarch64.neon.uqrshrn.v2i32(<2 x i64>, i32) nounwind readnone

define i32 @uqshrn1s(i64 %A) nounwind {
; CHECK-LABEL: uqshrn1s:
; CHECK: uqshrn {{s[0-9]+}}, d0, #1
  %tmp = call i32 @llvm.aarch64.neon.uqshrn.i32(i64 %A, i32 1)
  ret i32 %tmp
}

define <8 x i8> @uqshrn8b(<8 x i16>* %A) nounwind {
;CHECK-LABEL: uqshrn8b:
;CHECK: uqshrn.8b v0, {{v[0-9]+}}, #1
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp3 = call <8 x i8> @llvm.aarch64.neon.uqshrn.v8i8(<8 x i16> %tmp1, i32 1)
        ret <8 x i8> %tmp3
}

define <4 x i16> @uqshrn4h(<4 x i32>* %A) nounwind {
;CHECK-LABEL: uqshrn4h:
;CHECK: uqshrn.4h v0, {{v[0-9]+}}, #1
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp3 = call <4 x i16> @llvm.aarch64.neon.uqshrn.v4i16(<4 x i32> %tmp1, i32 1)
        ret <4 x i16> %tmp3
}

define <2 x i32> @uqshrn2s(<2 x i64>* %A) nounwind {
;CHECK-LABEL: uqshrn2s:
;CHECK: uqshrn.2s v0, {{v[0-9]+}}, #1
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp3 = call <2 x i32> @llvm.aarch64.neon.uqshrn.v2i32(<2 x i64> %tmp1, i32 1)
        ret <2 x i32> %tmp3
}

define <16 x i8> @uqshrn16b(<8 x i8>* %ret, <8 x i16>* %A) nounwind {
;CHECK-LABEL: uqshrn16b:
;CHECK: uqshrn2.16b v0, {{v[0-9]+}}, #1
        %out = load <8 x i8>, <8 x i8>* %ret
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp3 = call <8 x i8> @llvm.aarch64.neon.uqshrn.v8i8(<8 x i16> %tmp1, i32 1)
        %tmp4 = shufflevector <8 x i8> %out, <8 x i8> %tmp3, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
        ret <16 x i8> %tmp4
}

define <8 x i16> @uqshrn8h(<4 x i16>* %ret, <4 x i32>* %A) nounwind {
;CHECK-LABEL: uqshrn8h:
;CHECK: uqshrn2.8h v0, {{v[0-9]+}}, #1
  %out = load <4 x i16>, <4 x i16>* %ret
  %tmp1 = load <4 x i32>, <4 x i32>* %A
  %tmp3 = call <4 x i16> @llvm.aarch64.neon.uqshrn.v4i16(<4 x i32> %tmp1, i32 1)
  %tmp4 = shufflevector <4 x i16> %out, <4 x i16> %tmp3, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i16> %tmp4
}

define <4 x i32> @uqshrn4s(<2 x i32>* %ret, <2 x i64>* %A) nounwind {
;CHECK-LABEL: uqshrn4s:
;CHECK: uqshrn2.4s v0, {{v[0-9]+}}, #1
  %out = load <2 x i32>, <2 x i32>* %ret
  %tmp1 = load <2 x i64>, <2 x i64>* %A
  %tmp3 = call <2 x i32> @llvm.aarch64.neon.uqshrn.v2i32(<2 x i64> %tmp1, i32 1)
  %tmp4 = shufflevector <2 x i32> %out, <2 x i32> %tmp3, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x i32> %tmp4
}

declare i32  @llvm.aarch64.neon.uqshrn.i32(i64, i32) nounwind readnone
declare <8 x i8>  @llvm.aarch64.neon.uqshrn.v8i8(<8 x i16>, i32) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.uqshrn.v4i16(<4 x i32>, i32) nounwind readnone
declare <2 x i32> @llvm.aarch64.neon.uqshrn.v2i32(<2 x i64>, i32) nounwind readnone

define <8 x i16> @ushll8h(<8 x i8>* %A) nounwind {
;CHECK-LABEL: ushll8h:
;CHECK: ushll.8h v0, {{v[0-9]+}}, #1
        %tmp1 = load <8 x i8>, <8 x i8>* %A
        %tmp2 = zext <8 x i8> %tmp1 to <8 x i16>
        %tmp3 = shl <8 x i16> %tmp2, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
        ret <8 x i16> %tmp3
}

define <4 x i32> @ushll4s(<4 x i16>* %A) nounwind {
;CHECK-LABEL: ushll4s:
;CHECK: ushll.4s v0, {{v[0-9]+}}, #1
        %tmp1 = load <4 x i16>, <4 x i16>* %A
        %tmp2 = zext <4 x i16> %tmp1 to <4 x i32>
        %tmp3 = shl <4 x i32> %tmp2, <i32 1, i32 1, i32 1, i32 1>
        ret <4 x i32> %tmp3
}

define <2 x i64> @ushll2d(<2 x i32>* %A) nounwind {
;CHECK-LABEL: ushll2d:
;CHECK: ushll.2d v0, {{v[0-9]+}}, #1
        %tmp1 = load <2 x i32>, <2 x i32>* %A
        %tmp2 = zext <2 x i32> %tmp1 to <2 x i64>
        %tmp3 = shl <2 x i64> %tmp2, <i64 1, i64 1>
        ret <2 x i64> %tmp3
}

define <8 x i16> @ushll2_8h(<16 x i8>* %A) nounwind {
;CHECK-LABEL: ushll2_8h:
;CHECK: ushll.8h v0, {{v[0-9]+}}, #1
        %load1 = load <16 x i8>, <16 x i8>* %A
        %tmp1 = shufflevector <16 x i8> %load1, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
        %tmp2 = zext <8 x i8> %tmp1 to <8 x i16>
        %tmp3 = shl <8 x i16> %tmp2, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
        ret <8 x i16> %tmp3
}

define <4 x i32> @ushll2_4s(<8 x i16>* %A) nounwind {
;CHECK-LABEL: ushll2_4s:
;CHECK: ushll.4s v0, {{v[0-9]+}}, #1
        %load1 = load <8 x i16>, <8 x i16>* %A
        %tmp1 = shufflevector <8 x i16> %load1, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
        %tmp2 = zext <4 x i16> %tmp1 to <4 x i32>
        %tmp3 = shl <4 x i32> %tmp2, <i32 1, i32 1, i32 1, i32 1>
        ret <4 x i32> %tmp3
}

define <2 x i64> @ushll2_2d(<4 x i32>* %A) nounwind {
;CHECK-LABEL: ushll2_2d:
;CHECK: ushll.2d v0, {{v[0-9]+}}, #1
        %load1 = load <4 x i32>, <4 x i32>* %A
        %tmp1 = shufflevector <4 x i32> %load1, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
        %tmp2 = zext <2 x i32> %tmp1 to <2 x i64>
        %tmp3 = shl <2 x i64> %tmp2, <i64 1, i64 1>
        ret <2 x i64> %tmp3
}

define <8 x i16> @sshll8h(<8 x i8>* %A) nounwind {
;CHECK-LABEL: sshll8h:
;CHECK: sshll.8h v0, {{v[0-9]+}}, #1
        %tmp1 = load <8 x i8>, <8 x i8>* %A
        %tmp2 = sext <8 x i8> %tmp1 to <8 x i16>
        %tmp3 = shl <8 x i16> %tmp2, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
        ret <8 x i16> %tmp3
}

define <4 x i32> @sshll4s(<4 x i16>* %A) nounwind {
;CHECK-LABEL: sshll4s:
;CHECK: sshll.4s v0, {{v[0-9]+}}, #1
        %tmp1 = load <4 x i16>, <4 x i16>* %A
        %tmp2 = sext <4 x i16> %tmp1 to <4 x i32>
        %tmp3 = shl <4 x i32> %tmp2, <i32 1, i32 1, i32 1, i32 1>
        ret <4 x i32> %tmp3
}

define <2 x i64> @sshll2d(<2 x i32>* %A) nounwind {
;CHECK-LABEL: sshll2d:
;CHECK: sshll.2d v0, {{v[0-9]+}}, #1
        %tmp1 = load <2 x i32>, <2 x i32>* %A
        %tmp2 = sext <2 x i32> %tmp1 to <2 x i64>
        %tmp3 = shl <2 x i64> %tmp2, <i64 1, i64 1>
        ret <2 x i64> %tmp3
}

define <8 x i16> @sshll2_8h(<16 x i8>* %A) nounwind {
;CHECK-LABEL: sshll2_8h:
;CHECK: sshll.8h v0, {{v[0-9]+}}, #1
        %load1 = load <16 x i8>, <16 x i8>* %A
        %tmp1 = shufflevector <16 x i8> %load1, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
        %tmp2 = sext <8 x i8> %tmp1 to <8 x i16>
        %tmp3 = shl <8 x i16> %tmp2, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
        ret <8 x i16> %tmp3
}

define <4 x i32> @sshll2_4s(<8 x i16>* %A) nounwind {
;CHECK-LABEL: sshll2_4s:
;CHECK: sshll.4s v0, {{v[0-9]+}}, #1
        %load1 = load <8 x i16>, <8 x i16>* %A
        %tmp1 = shufflevector <8 x i16> %load1, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
        %tmp2 = sext <4 x i16> %tmp1 to <4 x i32>
        %tmp3 = shl <4 x i32> %tmp2, <i32 1, i32 1, i32 1, i32 1>
        ret <4 x i32> %tmp3
}

define <2 x i64> @sshll2_2d(<4 x i32>* %A) nounwind {
;CHECK-LABEL: sshll2_2d:
;CHECK: sshll.2d v0, {{v[0-9]+}}, #1
        %load1 = load <4 x i32>, <4 x i32>* %A
        %tmp1 = shufflevector <4 x i32> %load1, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
        %tmp2 = sext <2 x i32> %tmp1 to <2 x i64>
        %tmp3 = shl <2 x i64> %tmp2, <i64 1, i64 1>
        ret <2 x i64> %tmp3
}

define <8 x i8> @sqshli8b(<8 x i8>* %A) nounwind {
;CHECK-LABEL: sqshli8b:
;CHECK: sqshl.8b v0, {{v[0-9]+}}, #1
        %tmp1 = load <8 x i8>, <8 x i8>* %A
        %tmp3 = call <8 x i8> @llvm.aarch64.neon.sqshl.v8i8(<8 x i8> %tmp1, <8 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>)
        ret <8 x i8> %tmp3
}

define <4 x i16> @sqshli4h(<4 x i16>* %A) nounwind {
;CHECK-LABEL: sqshli4h:
;CHECK: sqshl.4h v0, {{v[0-9]+}}, #1
        %tmp1 = load <4 x i16>, <4 x i16>* %A
        %tmp3 = call <4 x i16> @llvm.aarch64.neon.sqshl.v4i16(<4 x i16> %tmp1, <4 x i16> <i16 1, i16 1, i16 1, i16 1>)
        ret <4 x i16> %tmp3
}

define <2 x i32> @sqshli2s(<2 x i32>* %A) nounwind {
;CHECK-LABEL: sqshli2s:
;CHECK: sqshl.2s v0, {{v[0-9]+}}, #1
        %tmp1 = load <2 x i32>, <2 x i32>* %A
        %tmp3 = call <2 x i32> @llvm.aarch64.neon.sqshl.v2i32(<2 x i32> %tmp1, <2 x i32> <i32 1, i32 1>)
        ret <2 x i32> %tmp3
}

define <16 x i8> @sqshli16b(<16 x i8>* %A) nounwind {
;CHECK-LABEL: sqshli16b:
;CHECK: sqshl.16b v0, {{v[0-9]+}}, #1
        %tmp1 = load <16 x i8>, <16 x i8>* %A
        %tmp3 = call <16 x i8> @llvm.aarch64.neon.sqshl.v16i8(<16 x i8> %tmp1, <16 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>)
        ret <16 x i8> %tmp3
}

define <8 x i16> @sqshli8h(<8 x i16>* %A) nounwind {
;CHECK-LABEL: sqshli8h:
;CHECK: sqshl.8h v0, {{v[0-9]+}}, #1
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp3 = call <8 x i16> @llvm.aarch64.neon.sqshl.v8i16(<8 x i16> %tmp1, <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>)
        ret <8 x i16> %tmp3
}

define <4 x i32> @sqshli4s(<4 x i32>* %A) nounwind {
;CHECK-LABEL: sqshli4s:
;CHECK: sqshl.4s v0, {{v[0-9]+}}, #1
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp3 = call <4 x i32> @llvm.aarch64.neon.sqshl.v4i32(<4 x i32> %tmp1, <4 x i32> <i32 1, i32 1, i32 1, i32 1>)
        ret <4 x i32> %tmp3
}

define <2 x i64> @sqshli2d(<2 x i64>* %A) nounwind {
;CHECK-LABEL: sqshli2d:
;CHECK: sqshl.2d v0, {{v[0-9]+}}, #1
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp3 = call <2 x i64> @llvm.aarch64.neon.sqshl.v2i64(<2 x i64> %tmp1, <2 x i64> <i64 1, i64 1>)
        ret <2 x i64> %tmp3
}

define <8 x i8> @uqshli8b(<8 x i8>* %A) nounwind {
;CHECK-LABEL: uqshli8b:
;CHECK: uqshl.8b v0, {{v[0-9]+}}, #1
        %tmp1 = load <8 x i8>, <8 x i8>* %A
        %tmp3 = call <8 x i8> @llvm.aarch64.neon.uqshl.v8i8(<8 x i8> %tmp1, <8 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>)
        ret <8 x i8> %tmp3
}

define <8 x i8> @uqshli8b_1(<8 x i8>* %A) nounwind {
;CHECK-LABEL: uqshli8b_1:
;CHECK: movi.8b [[REG:v[0-9]+]], #8
;CHECK: uqshl.8b v0, v0, [[REG]]
        %tmp1 = load <8 x i8>, <8 x i8>* %A
        %tmp3 = call <8 x i8> @llvm.aarch64.neon.uqshl.v8i8(<8 x i8> %tmp1, <8 x i8> <i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8>)
        ret <8 x i8> %tmp3
}

define <4 x i16> @uqshli4h(<4 x i16>* %A) nounwind {
;CHECK-LABEL: uqshli4h:
;CHECK: uqshl.4h v0, {{v[0-9]+}}, #1
        %tmp1 = load <4 x i16>, <4 x i16>* %A
        %tmp3 = call <4 x i16> @llvm.aarch64.neon.uqshl.v4i16(<4 x i16> %tmp1, <4 x i16> <i16 1, i16 1, i16 1, i16 1>)
        ret <4 x i16> %tmp3
}

define <2 x i32> @uqshli2s(<2 x i32>* %A) nounwind {
;CHECK-LABEL: uqshli2s:
;CHECK: uqshl.2s v0, {{v[0-9]+}}, #1
        %tmp1 = load <2 x i32>, <2 x i32>* %A
        %tmp3 = call <2 x i32> @llvm.aarch64.neon.uqshl.v2i32(<2 x i32> %tmp1, <2 x i32> <i32 1, i32 1>)
        ret <2 x i32> %tmp3
}

define <16 x i8> @uqshli16b(<16 x i8>* %A) nounwind {
;CHECK-LABEL: uqshli16b:
;CHECK: uqshl.16b
        %tmp1 = load <16 x i8>, <16 x i8>* %A
        %tmp3 = call <16 x i8> @llvm.aarch64.neon.uqshl.v16i8(<16 x i8> %tmp1, <16 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>)
        ret <16 x i8> %tmp3
}

define <8 x i16> @uqshli8h(<8 x i16>* %A) nounwind {
;CHECK-LABEL: uqshli8h:
;CHECK: uqshl.8h v0, {{v[0-9]+}}, #1
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp3 = call <8 x i16> @llvm.aarch64.neon.uqshl.v8i16(<8 x i16> %tmp1, <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>)
        ret <8 x i16> %tmp3
}

define <4 x i32> @uqshli4s(<4 x i32>* %A) nounwind {
;CHECK-LABEL: uqshli4s:
;CHECK: uqshl.4s v0, {{v[0-9]+}}, #1
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp3 = call <4 x i32> @llvm.aarch64.neon.uqshl.v4i32(<4 x i32> %tmp1, <4 x i32> <i32 1, i32 1, i32 1, i32 1>)
        ret <4 x i32> %tmp3
}

define <2 x i64> @uqshli2d(<2 x i64>* %A) nounwind {
;CHECK-LABEL: uqshli2d:
;CHECK: uqshl.2d v0, {{v[0-9]+}}, #1
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp3 = call <2 x i64> @llvm.aarch64.neon.uqshl.v2i64(<2 x i64> %tmp1, <2 x i64> <i64 1, i64 1>)
        ret <2 x i64> %tmp3
}

define <8 x i8> @ursra8b(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: ursra8b:
;CHECK: ursra.8b v0, {{v[0-9]+}}, #1
        %tmp1 = load <8 x i8>, <8 x i8>* %A
        %tmp3 = call <8 x i8> @llvm.aarch64.neon.urshl.v8i8(<8 x i8> %tmp1, <8 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>)
        %tmp4 = load <8 x i8>, <8 x i8>* %B
        %tmp5 = add <8 x i8> %tmp3, %tmp4
        ret <8 x i8> %tmp5
}

define <4 x i16> @ursra4h(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: ursra4h:
;CHECK: ursra.4h v0, {{v[0-9]+}}, #1
        %tmp1 = load <4 x i16>, <4 x i16>* %A
        %tmp3 = call <4 x i16> @llvm.aarch64.neon.urshl.v4i16(<4 x i16> %tmp1, <4 x i16> <i16 -1, i16 -1, i16 -1, i16 -1>)
        %tmp4 = load <4 x i16>, <4 x i16>* %B
        %tmp5 = add <4 x i16> %tmp3, %tmp4
        ret <4 x i16> %tmp5
}

define <2 x i32> @ursra2s(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: ursra2s:
;CHECK: ursra.2s v0, {{v[0-9]+}}, #1
        %tmp1 = load <2 x i32>, <2 x i32>* %A
        %tmp3 = call <2 x i32> @llvm.aarch64.neon.urshl.v2i32(<2 x i32> %tmp1, <2 x i32> <i32 -1, i32 -1>)
        %tmp4 = load <2 x i32>, <2 x i32>* %B
        %tmp5 = add <2 x i32> %tmp3, %tmp4
        ret <2 x i32> %tmp5
}

define <16 x i8> @ursra16b(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: ursra16b:
;CHECK: ursra.16b v0, {{v[0-9]+}}, #1
        %tmp1 = load <16 x i8>, <16 x i8>* %A
        %tmp3 = call <16 x i8> @llvm.aarch64.neon.urshl.v16i8(<16 x i8> %tmp1, <16 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>)
        %tmp4 = load <16 x i8>, <16 x i8>* %B
        %tmp5 = add <16 x i8> %tmp3, %tmp4
         ret <16 x i8> %tmp5
}

define <8 x i16> @ursra8h(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: ursra8h:
;CHECK: ursra.8h v0, {{v[0-9]+}}, #1
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp3 = call <8 x i16> @llvm.aarch64.neon.urshl.v8i16(<8 x i16> %tmp1, <8 x i16> <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>)
        %tmp4 = load <8 x i16>, <8 x i16>* %B
        %tmp5 = add <8 x i16> %tmp3, %tmp4
         ret <8 x i16> %tmp5
}

define <4 x i32> @ursra4s(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: ursra4s:
;CHECK: ursra.4s v0, {{v[0-9]+}}, #1
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp3 = call <4 x i32> @llvm.aarch64.neon.urshl.v4i32(<4 x i32> %tmp1, <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>)
        %tmp4 = load <4 x i32>, <4 x i32>* %B
        %tmp5 = add <4 x i32> %tmp3, %tmp4
         ret <4 x i32> %tmp5
}

define <2 x i64> @ursra2d(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK-LABEL: ursra2d:
;CHECK: ursra.2d v0, {{v[0-9]+}}, #1
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp3 = call <2 x i64> @llvm.aarch64.neon.urshl.v2i64(<2 x i64> %tmp1, <2 x i64> <i64 -1, i64 -1>)
        %tmp4 = load <2 x i64>, <2 x i64>* %B
        %tmp5 = add <2 x i64> %tmp3, %tmp4
         ret <2 x i64> %tmp5
}

define <8 x i8> @srsra8b(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: srsra8b:
;CHECK: srsra.8b v0, {{v[0-9]+}}, #1
        %tmp1 = load <8 x i8>, <8 x i8>* %A
        %tmp3 = call <8 x i8> @llvm.aarch64.neon.srshl.v8i8(<8 x i8> %tmp1, <8 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>)
        %tmp4 = load <8 x i8>, <8 x i8>* %B
        %tmp5 = add <8 x i8> %tmp3, %tmp4
        ret <8 x i8> %tmp5
}

define <4 x i16> @srsra4h(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: srsra4h:
;CHECK: srsra.4h v0, {{v[0-9]+}}, #1
        %tmp1 = load <4 x i16>, <4 x i16>* %A
        %tmp3 = call <4 x i16> @llvm.aarch64.neon.srshl.v4i16(<4 x i16> %tmp1, <4 x i16> <i16 -1, i16 -1, i16 -1, i16 -1>)
        %tmp4 = load <4 x i16>, <4 x i16>* %B
        %tmp5 = add <4 x i16> %tmp3, %tmp4
        ret <4 x i16> %tmp5
}

define <2 x i32> @srsra2s(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: srsra2s:
;CHECK: srsra.2s v0, {{v[0-9]+}}, #1
        %tmp1 = load <2 x i32>, <2 x i32>* %A
        %tmp3 = call <2 x i32> @llvm.aarch64.neon.srshl.v2i32(<2 x i32> %tmp1, <2 x i32> <i32 -1, i32 -1>)
        %tmp4 = load <2 x i32>, <2 x i32>* %B
        %tmp5 = add <2 x i32> %tmp3, %tmp4
        ret <2 x i32> %tmp5
}

define <16 x i8> @srsra16b(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: srsra16b:
;CHECK: srsra.16b v0, {{v[0-9]+}}, #1
        %tmp1 = load <16 x i8>, <16 x i8>* %A
        %tmp3 = call <16 x i8> @llvm.aarch64.neon.srshl.v16i8(<16 x i8> %tmp1, <16 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>)
        %tmp4 = load <16 x i8>, <16 x i8>* %B
        %tmp5 = add <16 x i8> %tmp3, %tmp4
         ret <16 x i8> %tmp5
}

define <8 x i16> @srsra8h(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: srsra8h:
;CHECK: srsra.8h v0, {{v[0-9]+}}, #1
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp3 = call <8 x i16> @llvm.aarch64.neon.srshl.v8i16(<8 x i16> %tmp1, <8 x i16> <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>)
        %tmp4 = load <8 x i16>, <8 x i16>* %B
        %tmp5 = add <8 x i16> %tmp3, %tmp4
         ret <8 x i16> %tmp5
}

define <4 x i32> @srsra4s(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: srsra4s:
;CHECK: srsra.4s v0, {{v[0-9]+}}, #1
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp3 = call <4 x i32> @llvm.aarch64.neon.srshl.v4i32(<4 x i32> %tmp1, <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>)
        %tmp4 = load <4 x i32>, <4 x i32>* %B
        %tmp5 = add <4 x i32> %tmp3, %tmp4
         ret <4 x i32> %tmp5
}

define <2 x i64> @srsra2d(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK-LABEL: srsra2d:
;CHECK: srsra.2d v0, {{v[0-9]+}}, #1
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp3 = call <2 x i64> @llvm.aarch64.neon.srshl.v2i64(<2 x i64> %tmp1, <2 x i64> <i64 -1, i64 -1>)
        %tmp4 = load <2 x i64>, <2 x i64>* %B
        %tmp5 = add <2 x i64> %tmp3, %tmp4
         ret <2 x i64> %tmp5
}

define <8 x i8> @usra8b(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: usra8b:
;CHECK: usra.8b v0, {{v[0-9]+}}, #1
        %tmp1 = load <8 x i8>, <8 x i8>* %A
        %tmp3 = lshr <8 x i8> %tmp1, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
        %tmp4 = load <8 x i8>, <8 x i8>* %B
        %tmp5 = add <8 x i8> %tmp3, %tmp4
        ret <8 x i8> %tmp5
}

define <4 x i16> @usra4h(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: usra4h:
;CHECK: usra.4h v0, {{v[0-9]+}}, #1
        %tmp1 = load <4 x i16>, <4 x i16>* %A
        %tmp3 = lshr <4 x i16> %tmp1, <i16 1, i16 1, i16 1, i16 1>
        %tmp4 = load <4 x i16>, <4 x i16>* %B
        %tmp5 = add <4 x i16> %tmp3, %tmp4
        ret <4 x i16> %tmp5
}

define <2 x i32> @usra2s(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: usra2s:
;CHECK: usra.2s v0, {{v[0-9]+}}, #1
        %tmp1 = load <2 x i32>, <2 x i32>* %A
        %tmp3 = lshr <2 x i32> %tmp1, <i32 1, i32 1>
        %tmp4 = load <2 x i32>, <2 x i32>* %B
        %tmp5 = add <2 x i32> %tmp3, %tmp4
        ret <2 x i32> %tmp5
}

define <16 x i8> @usra16b(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: usra16b:
;CHECK: usra.16b v0, {{v[0-9]+}}, #1
        %tmp1 = load <16 x i8>, <16 x i8>* %A
        %tmp3 = lshr <16 x i8> %tmp1, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
        %tmp4 = load <16 x i8>, <16 x i8>* %B
        %tmp5 = add <16 x i8> %tmp3, %tmp4
         ret <16 x i8> %tmp5
}

define <8 x i16> @usra8h(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: usra8h:
;CHECK: usra.8h v0, {{v[0-9]+}}, #1
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp3 = lshr <8 x i16> %tmp1, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
        %tmp4 = load <8 x i16>, <8 x i16>* %B
        %tmp5 = add <8 x i16> %tmp3, %tmp4
         ret <8 x i16> %tmp5
}

define <4 x i32> @usra4s(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: usra4s:
;CHECK: usra.4s v0, {{v[0-9]+}}, #1
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp3 = lshr <4 x i32> %tmp1, <i32 1, i32 1, i32 1, i32 1>
        %tmp4 = load <4 x i32>, <4 x i32>* %B
        %tmp5 = add <4 x i32> %tmp3, %tmp4
         ret <4 x i32> %tmp5
}

define <2 x i64> @usra2d(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK-LABEL: usra2d:
;CHECK: usra.2d v0, {{v[0-9]+}}, #1
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp3 = lshr <2 x i64> %tmp1, <i64 1, i64 1>
        %tmp4 = load <2 x i64>, <2 x i64>* %B
        %tmp5 = add <2 x i64> %tmp3, %tmp4
         ret <2 x i64> %tmp5
}

define <8 x i8> @ssra8b(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: ssra8b:
;CHECK: ssra.8b v0, {{v[0-9]+}}, #1
        %tmp1 = load <8 x i8>, <8 x i8>* %A
        %tmp3 = ashr <8 x i8> %tmp1, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
        %tmp4 = load <8 x i8>, <8 x i8>* %B
        %tmp5 = add <8 x i8> %tmp3, %tmp4
        ret <8 x i8> %tmp5
}

define <4 x i16> @ssra4h(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: ssra4h:
;CHECK: ssra.4h v0, {{v[0-9]+}}, #1
        %tmp1 = load <4 x i16>, <4 x i16>* %A
        %tmp3 = ashr <4 x i16> %tmp1, <i16 1, i16 1, i16 1, i16 1>
        %tmp4 = load <4 x i16>, <4 x i16>* %B
        %tmp5 = add <4 x i16> %tmp3, %tmp4
        ret <4 x i16> %tmp5
}

define <2 x i32> @ssra2s(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: ssra2s:
;CHECK: ssra.2s v0, {{v[0-9]+}}, #1
        %tmp1 = load <2 x i32>, <2 x i32>* %A
        %tmp3 = ashr <2 x i32> %tmp1, <i32 1, i32 1>
        %tmp4 = load <2 x i32>, <2 x i32>* %B
        %tmp5 = add <2 x i32> %tmp3, %tmp4
        ret <2 x i32> %tmp5
}

define <16 x i8> @ssra16b(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: ssra16b:
;CHECK: ssra.16b v0, {{v[0-9]+}}, #1
        %tmp1 = load <16 x i8>, <16 x i8>* %A
        %tmp3 = ashr <16 x i8> %tmp1, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
        %tmp4 = load <16 x i8>, <16 x i8>* %B
        %tmp5 = add <16 x i8> %tmp3, %tmp4
         ret <16 x i8> %tmp5
}

define <8 x i16> @ssra8h(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: ssra8h:
;CHECK: ssra.8h v0, {{v[0-9]+}}, #1
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp3 = ashr <8 x i16> %tmp1, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
        %tmp4 = load <8 x i16>, <8 x i16>* %B
        %tmp5 = add <8 x i16> %tmp3, %tmp4
         ret <8 x i16> %tmp5
}

define <4 x i32> @ssra4s(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: ssra4s:
;CHECK: ssra.4s v0, {{v[0-9]+}}, #1
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp3 = ashr <4 x i32> %tmp1, <i32 1, i32 1, i32 1, i32 1>
        %tmp4 = load <4 x i32>, <4 x i32>* %B
        %tmp5 = add <4 x i32> %tmp3, %tmp4
         ret <4 x i32> %tmp5
}

define <2 x i64> @ssra2d(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK-LABEL: ssra2d:
;CHECK: ssra.2d v0, {{v[0-9]+}}, #1
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp3 = ashr <2 x i64> %tmp1, <i64 1, i64 1>
        %tmp4 = load <2 x i64>, <2 x i64>* %B
        %tmp5 = add <2 x i64> %tmp3, %tmp4
         ret <2 x i64> %tmp5
}

define <8 x i8> @shr_orr8b(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: shr_orr8b:
;CHECK: shr.8b v0, {{v[0-9]+}}, #1
;CHECK-NEXT: orr.8b
;CHECK-NEXT: ret
        %tmp1 = load <8 x i8>, <8 x i8>* %A
        %tmp4 = load <8 x i8>, <8 x i8>* %B
        %tmp3 = lshr <8 x i8> %tmp1, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
        %tmp5 = or <8 x i8> %tmp3, %tmp4
        ret <8 x i8> %tmp5
}

define <4 x i16> @shr_orr4h(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: shr_orr4h:
;CHECK: shr.4h v0, {{v[0-9]+}}, #1
;CHECK-NEXT: orr.8b
;CHECK-NEXT: ret
        %tmp1 = load <4 x i16>, <4 x i16>* %A
        %tmp4 = load <4 x i16>, <4 x i16>* %B
        %tmp3 = lshr <4 x i16> %tmp1, <i16 1, i16 1, i16 1, i16 1>
        %tmp5 = or <4 x i16> %tmp3, %tmp4
        ret <4 x i16> %tmp5
}

define <2 x i32> @shr_orr2s(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: shr_orr2s:
;CHECK: shr.2s v0, {{v[0-9]+}}, #1
;CHECK-NEXT: orr.8b
;CHECK-NEXT: ret
        %tmp1 = load <2 x i32>, <2 x i32>* %A
        %tmp4 = load <2 x i32>, <2 x i32>* %B
        %tmp3 = lshr <2 x i32> %tmp1, <i32 1, i32 1>
        %tmp5 = or <2 x i32> %tmp3, %tmp4
        ret <2 x i32> %tmp5
}

define <16 x i8> @shr_orr16b(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: shr_orr16b:
;CHECK: shr.16b v0, {{v[0-9]+}}, #1
;CHECK-NEXT: orr.16b
;CHECK-NEXT: ret
        %tmp1 = load <16 x i8>, <16 x i8>* %A
        %tmp4 = load <16 x i8>, <16 x i8>* %B
        %tmp3 = lshr <16 x i8> %tmp1, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
        %tmp5 = or <16 x i8> %tmp3, %tmp4
         ret <16 x i8> %tmp5
}

define <8 x i16> @shr_orr8h(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: shr_orr8h:
;CHECK: shr.8h v0, {{v[0-9]+}}, #1
;CHECK-NEXT: orr.16b
;CHECK-NEXT: ret
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp4 = load <8 x i16>, <8 x i16>* %B
        %tmp3 = lshr <8 x i16> %tmp1, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
        %tmp5 = or <8 x i16> %tmp3, %tmp4
         ret <8 x i16> %tmp5
}

define <4 x i32> @shr_orr4s(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: shr_orr4s:
;CHECK: shr.4s v0, {{v[0-9]+}}, #1
;CHECK-NEXT: orr.16b
;CHECK-NEXT: ret
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp4 = load <4 x i32>, <4 x i32>* %B
        %tmp3 = lshr <4 x i32> %tmp1, <i32 1, i32 1, i32 1, i32 1>
        %tmp5 = or <4 x i32> %tmp3, %tmp4
         ret <4 x i32> %tmp5
}

define <2 x i64> @shr_orr2d(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK-LABEL: shr_orr2d:
;CHECK: shr.2d v0, {{v[0-9]+}}, #1
;CHECK-NEXT: orr.16b
;CHECK-NEXT: ret
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp4 = load <2 x i64>, <2 x i64>* %B
        %tmp3 = lshr <2 x i64> %tmp1, <i64 1, i64 1>
        %tmp5 = or <2 x i64> %tmp3, %tmp4
         ret <2 x i64> %tmp5
}

define <8 x i8> @shl_orr8b(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: shl_orr8b:
;CHECK: shl.8b v0, {{v[0-9]+}}, #1
;CHECK-NEXT: orr.8b
;CHECK-NEXT: ret
        %tmp1 = load <8 x i8>, <8 x i8>* %A
        %tmp4 = load <8 x i8>, <8 x i8>* %B
        %tmp3 = shl <8 x i8> %tmp1, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
        %tmp5 = or <8 x i8> %tmp3, %tmp4
        ret <8 x i8> %tmp5
}

define <4 x i16> @shl_orr4h(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: shl_orr4h:
;CHECK: shl.4h v0, {{v[0-9]+}}, #1
;CHECK-NEXT: orr.8b
;CHECK-NEXT: ret
        %tmp1 = load <4 x i16>, <4 x i16>* %A
        %tmp4 = load <4 x i16>, <4 x i16>* %B
        %tmp3 = shl <4 x i16> %tmp1, <i16 1, i16 1, i16 1, i16 1>
        %tmp5 = or <4 x i16> %tmp3, %tmp4
        ret <4 x i16> %tmp5
}

define <2 x i32> @shl_orr2s(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: shl_orr2s:
;CHECK: shl.2s v0, {{v[0-9]+}}, #1
;CHECK-NEXT: orr.8b
;CHECK-NEXT: ret
        %tmp1 = load <2 x i32>, <2 x i32>* %A
        %tmp4 = load <2 x i32>, <2 x i32>* %B
        %tmp3 = shl <2 x i32> %tmp1, <i32 1, i32 1>
        %tmp5 = or <2 x i32> %tmp3, %tmp4
        ret <2 x i32> %tmp5
}

define <16 x i8> @shl_orr16b(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: shl_orr16b:
;CHECK: shl.16b v0, {{v[0-9]+}}, #1
;CHECK-NEXT: orr.16b
;CHECK-NEXT: ret
        %tmp1 = load <16 x i8>, <16 x i8>* %A
        %tmp4 = load <16 x i8>, <16 x i8>* %B
        %tmp3 = shl <16 x i8> %tmp1, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
        %tmp5 = or <16 x i8> %tmp3, %tmp4
         ret <16 x i8> %tmp5
}

define <8 x i16> @shl_orr8h(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: shl_orr8h:
;CHECK: shl.8h v0, {{v[0-9]+}}, #1
;CHECK-NEXT: orr.16b
;CHECK-NEXT: ret
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp4 = load <8 x i16>, <8 x i16>* %B
        %tmp3 = shl <8 x i16> %tmp1, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
        %tmp5 = or <8 x i16> %tmp3, %tmp4
         ret <8 x i16> %tmp5
}

define <4 x i32> @shl_orr4s(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: shl_orr4s:
;CHECK: shl.4s v0, {{v[0-9]+}}, #1
;CHECK-NEXT: orr.16b
;CHECK-NEXT: ret
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp4 = load <4 x i32>, <4 x i32>* %B
        %tmp3 = shl <4 x i32> %tmp1, <i32 1, i32 1, i32 1, i32 1>
        %tmp5 = or <4 x i32> %tmp3, %tmp4
         ret <4 x i32> %tmp5
}

define <2 x i64> @shl_orr2d(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK-LABEL: shl_orr2d:
;CHECK: shl.2d v0, {{v[0-9]+}}, #1
;CHECK-NEXT: orr.16b
;CHECK-NEXT: ret
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp4 = load <2 x i64>, <2 x i64>* %B
        %tmp3 = shl <2 x i64> %tmp1, <i64 1, i64 1>
        %tmp5 = or <2 x i64> %tmp3, %tmp4
         ret <2 x i64> %tmp5
}

define <8 x i16> @shll(<8 x i8> %in) {
; CHECK-LABEL: shll:
; CHECK: shll.8h v0, {{v[0-9]+}}, #8
  %ext = zext <8 x i8> %in to <8 x i16>
  %res = shl <8 x i16> %ext, <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
  ret <8 x i16> %res
}

define <4 x i32> @shll_high(<8 x i16> %in) {
; CHECK-LABEL: shll_high
; CHECK: shll2.4s v0, {{v[0-9]+}}, #16
  %extract = shufflevector <8 x i16> %in, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %ext = zext <4 x i16> %extract to <4 x i32>
  %res = shl <4 x i32> %ext, <i32 16, i32 16, i32 16, i32 16>
  ret <4 x i32> %res
}

define <8 x i8> @sli8b(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: sli8b:
;CHECK: sli.8b v0, {{v[0-9]+}}, #1
        %tmp1 = load <8 x i8>, <8 x i8>* %A
        %tmp2 = load <8 x i8>, <8 x i8>* %B
        %tmp3 = call <8 x i8> @llvm.aarch64.neon.vsli.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2, i32 1)
        ret <8 x i8> %tmp3
}

define <4 x i16> @sli4h(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: sli4h:
;CHECK: sli.4h v0, {{v[0-9]+}}, #1
        %tmp1 = load <4 x i16>, <4 x i16>* %A
        %tmp2 = load <4 x i16>, <4 x i16>* %B
        %tmp3 = call <4 x i16> @llvm.aarch64.neon.vsli.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2, i32 1)
        ret <4 x i16> %tmp3
}

define <2 x i32> @sli2s(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: sli2s:
;CHECK: sli.2s v0, {{v[0-9]+}}, #1
        %tmp1 = load <2 x i32>, <2 x i32>* %A
        %tmp2 = load <2 x i32>, <2 x i32>* %B
        %tmp3 = call <2 x i32> @llvm.aarch64.neon.vsli.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2, i32 1)
        ret <2 x i32> %tmp3
}

define <1 x i64> @sli1d(<1 x i64>* %A, <1 x i64>* %B) nounwind {
;CHECK-LABEL: sli1d:
;CHECK: sli d0, {{d[0-9]+}}, #1
        %tmp1 = load <1 x i64>, <1 x i64>* %A
        %tmp2 = load <1 x i64>, <1 x i64>* %B
        %tmp3 = call <1 x i64> @llvm.aarch64.neon.vsli.v1i64(<1 x i64> %tmp1, <1 x i64> %tmp2, i32 1)
        ret <1 x i64> %tmp3
}

define <16 x i8> @sli16b(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: sli16b:
;CHECK: sli.16b v0, {{v[0-9]+}}, #1
        %tmp1 = load <16 x i8>, <16 x i8>* %A
        %tmp2 = load <16 x i8>, <16 x i8>* %B
        %tmp3 = call <16 x i8> @llvm.aarch64.neon.vsli.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2, i32 1)
        ret <16 x i8> %tmp3
}

define <8 x i16> @sli8h(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: sli8h:
;CHECK: sli.8h v0, {{v[0-9]+}}, #1
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp2 = load <8 x i16>, <8 x i16>* %B
        %tmp3 = call <8 x i16> @llvm.aarch64.neon.vsli.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2, i32 1)
        ret <8 x i16> %tmp3
}

define <4 x i32> @sli4s(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: sli4s:
;CHECK: sli.4s v0, {{v[0-9]+}}, #1
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp2 = load <4 x i32>, <4 x i32>* %B
        %tmp3 = call <4 x i32> @llvm.aarch64.neon.vsli.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2, i32 1)
        ret <4 x i32> %tmp3
}

define <2 x i64> @sli2d(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK-LABEL: sli2d:
;CHECK: sli.2d v0, {{v[0-9]+}}, #1
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp2 = load <2 x i64>, <2 x i64>* %B
        %tmp3 = call <2 x i64> @llvm.aarch64.neon.vsli.v2i64(<2 x i64> %tmp1, <2 x i64> %tmp2, i32 1)
        ret <2 x i64> %tmp3
}

declare <8 x i8>  @llvm.aarch64.neon.vsli.v8i8(<8 x i8>, <8 x i8>, i32) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.vsli.v4i16(<4 x i16>, <4 x i16>, i32) nounwind readnone
declare <2 x i32> @llvm.aarch64.neon.vsli.v2i32(<2 x i32>, <2 x i32>, i32) nounwind readnone
declare <1 x i64> @llvm.aarch64.neon.vsli.v1i64(<1 x i64>, <1 x i64>, i32) nounwind readnone

declare <16 x i8> @llvm.aarch64.neon.vsli.v16i8(<16 x i8>, <16 x i8>, i32) nounwind readnone
declare <8 x i16> @llvm.aarch64.neon.vsli.v8i16(<8 x i16>, <8 x i16>, i32) nounwind readnone
declare <4 x i32> @llvm.aarch64.neon.vsli.v4i32(<4 x i32>, <4 x i32>, i32) nounwind readnone
declare <2 x i64> @llvm.aarch64.neon.vsli.v2i64(<2 x i64>, <2 x i64>, i32) nounwind readnone

define <1 x i64> @ashr_v1i64(<1 x i64> %a, <1 x i64> %b) {
; CHECK-LABEL: ashr_v1i64:
; CHECK: neg d{{[0-9]+}}, d{{[0-9]+}}
; CHECK: sshl d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
  %c = ashr <1 x i64> %a, %b
  ret <1 x i64> %c
}
