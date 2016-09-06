; RUN: llc < %s -mtriple=arm64-apple-ios -mcpu=cyclone | FileCheck %s

; The non-byte ones used to fail with "Cannot select"

; CHECK-LABEL: ctpopv8i8
; CHECK: cnt.8b
define <8 x i8> @ctpopv8i8(<8 x i8> %x) nounwind readnone {
  %cnt = tail call <8 x i8> @llvm.ctpop.v8i8(<8 x i8> %x)
  ret <8 x i8> %cnt
}

declare <8 x i8> @llvm.ctpop.v8i8(<8 x i8>) nounwind readnone

; CHECK-LABEL: ctpopv4i16
; CHECK: cnt.8b
define <4 x i16> @ctpopv4i16(<4 x i16> %x) nounwind readnone {
  %cnt = tail call <4 x i16> @llvm.ctpop.v4i16(<4 x i16> %x)
  ret <4 x i16> %cnt
}

declare <4 x i16> @llvm.ctpop.v4i16(<4 x i16>) nounwind readnone

; CHECK-LABEL: ctpopv2i32
; CHECK: cnt.8b
define <2 x i32> @ctpopv2i32(<2 x i32> %x) nounwind readnone {
  %cnt = tail call <2 x i32> @llvm.ctpop.v2i32(<2 x i32> %x)
  ret <2 x i32> %cnt
}

declare <2 x i32> @llvm.ctpop.v2i32(<2 x i32>) nounwind readnone


; CHECK-LABEL: ctpopv16i8
; CHECK: cnt.16b
define <16 x i8> @ctpopv16i8(<16 x i8> %x) nounwind readnone {
  %cnt = tail call <16 x i8> @llvm.ctpop.v16i8(<16 x i8> %x)
  ret <16 x i8> %cnt
}

declare <16 x i8> @llvm.ctpop.v16i8(<16 x i8>) nounwind readnone

; CHECK-LABEL: ctpopv8i16
; CHECK: cnt.8b
define <8 x i16> @ctpopv8i16(<8 x i16> %x) nounwind readnone {
  %cnt = tail call <8 x i16> @llvm.ctpop.v8i16(<8 x i16> %x)
  ret <8 x i16> %cnt
}

declare <8 x i16> @llvm.ctpop.v8i16(<8 x i16>) nounwind readnone

; CHECK-LABEL: ctpopv4i32
; CHECK: cnt.8b
define <4 x i32> @ctpopv4i32(<4 x i32> %x) nounwind readnone {
  %cnt = tail call <4 x i32> @llvm.ctpop.v4i32(<4 x i32> %x)
  ret <4 x i32> %cnt
}

declare <4 x i32> @llvm.ctpop.v4i32(<4 x i32>) nounwind readnone

; CHECK-LABEL: ctpopv2i64
; CHECK: cnt.8b
define <2 x i64> @ctpopv2i64(<2 x i64> %x) nounwind readnone {
  %cnt = tail call <2 x i64> @llvm.ctpop.v2i64(<2 x i64> %x)
  ret <2 x i64> %cnt
}

declare <2 x i64> @llvm.ctpop.v2i64(<2 x i64>) nounwind readnone
