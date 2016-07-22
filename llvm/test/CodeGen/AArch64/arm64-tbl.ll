; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple | FileCheck %s

define <8 x i8> @tbl1_8b(<16 x i8> %A, <8 x i8> %B) nounwind {
; CHECK: tbl1_8b
; CHECK: tbl.8b
  %tmp3 = call <8 x i8> @llvm.aarch64.neon.tbl1.v8i8(<16 x i8> %A, <8 x i8> %B)
  ret <8 x i8> %tmp3
}

define <16 x i8> @tbl1_16b(<16 x i8> %A, <16 x i8> %B) nounwind {
; CHECK: tbl1_16b
; CHECK: tbl.16b
  %tmp3 = call <16 x i8> @llvm.aarch64.neon.tbl1.v16i8(<16 x i8> %A, <16 x i8> %B)
  ret <16 x i8> %tmp3
}

define <8 x i8> @tbl2_8b(<16 x i8> %A, <16 x i8> %B, <8 x i8> %C) {
; CHECK: tbl2_8b
; CHECK: tbl.8b
  %tmp3 = call <8 x i8> @llvm.aarch64.neon.tbl2.v8i8(<16 x i8> %A, <16 x i8> %B, <8 x i8> %C)
  ret <8 x i8> %tmp3
}

define <16 x i8> @tbl2_16b(<16 x i8> %A, <16 x i8> %B, <16 x i8> %C) {
; CHECK: tbl2_16b
; CHECK: tbl.16b
  %tmp3 = call <16 x i8> @llvm.aarch64.neon.tbl2.v16i8(<16 x i8> %A, <16 x i8> %B, <16 x i8> %C)
  ret <16 x i8> %tmp3
}

define <8 x i8> @tbl3_8b(<16 x i8> %A, <16 x i8> %B, <16 x i8> %C, <8 x i8> %D) {
; CHECK: tbl3_8b
; CHECK: tbl.8b
  %tmp3 = call <8 x i8> @llvm.aarch64.neon.tbl3.v8i8(<16 x i8> %A, <16 x i8> %B, <16 x i8> %C, <8 x i8> %D)
  ret <8 x i8> %tmp3
}

define <16 x i8> @tbl3_16b(<16 x i8> %A, <16 x i8> %B, <16 x i8> %C, <16 x i8> %D) {
; CHECK: tbl3_16b
; CHECK: tbl.16b
  %tmp3 = call <16 x i8> @llvm.aarch64.neon.tbl3.v16i8(<16 x i8> %A, <16 x i8> %B, <16 x i8> %C, <16 x i8> %D)
  ret <16 x i8> %tmp3
}

define <8 x i8> @tbl4_8b(<16 x i8> %A, <16 x i8> %B, <16 x i8> %C, <16 x i8> %D, <8 x i8> %E) {
; CHECK: tbl4_8b
; CHECK: tbl.8b
  %tmp3 = call <8 x i8> @llvm.aarch64.neon.tbl4.v8i8(<16 x i8> %A, <16 x i8> %B, <16 x i8> %C, <16 x i8> %D, <8 x i8> %E)
  ret <8 x i8> %tmp3
}

define <16 x i8> @tbl4_16b(<16 x i8> %A, <16 x i8> %B, <16 x i8> %C, <16 x i8> %D, <16 x i8> %E) {
; CHECK: tbl4_16b
; CHECK: tbl.16b
  %tmp3 = call <16 x i8> @llvm.aarch64.neon.tbl4.v16i8(<16 x i8> %A, <16 x i8> %B, <16 x i8> %C, <16 x i8> %D, <16 x i8> %E)
  ret <16 x i8> %tmp3
}

declare <8 x i8> @llvm.aarch64.neon.tbl1.v8i8(<16 x i8>, <8 x i8>) nounwind readnone
declare <16 x i8> @llvm.aarch64.neon.tbl1.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i8> @llvm.aarch64.neon.tbl2.v8i8(<16 x i8>, <16 x i8>, <8 x i8>) nounwind readnone
declare <16 x i8> @llvm.aarch64.neon.tbl2.v16i8(<16 x i8>, <16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i8> @llvm.aarch64.neon.tbl3.v8i8(<16 x i8>, <16 x i8>, <16 x i8>, <8 x i8>) nounwind readnone
declare <16 x i8> @llvm.aarch64.neon.tbl3.v16i8(<16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i8> @llvm.aarch64.neon.tbl4.v8i8(<16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <8 x i8>) nounwind readnone
declare <16 x i8> @llvm.aarch64.neon.tbl4.v16i8(<16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>) nounwind readnone

define <8 x i8> @tbx1_8b(<8 x i8> %A, <16 x i8> %B, <8 x i8> %C) nounwind {
; CHECK: tbx1_8b
; CHECK: tbx.8b
  %tmp3 = call <8 x i8> @llvm.aarch64.neon.tbx1.v8i8(<8 x i8> %A, <16 x i8> %B, <8 x i8> %C)
  ret <8 x i8> %tmp3
}

define <16 x i8> @tbx1_16b(<16 x i8> %A, <16 x i8> %B, <16 x i8> %C) nounwind {
; CHECK: tbx1_16b
; CHECK: tbx.16b
  %tmp3 = call <16 x i8> @llvm.aarch64.neon.tbx1.v16i8(<16 x i8> %A, <16 x i8> %B, <16 x i8> %C)
  ret <16 x i8> %tmp3
}

define <8 x i8> @tbx2_8b(<8 x i8> %A, <16 x i8> %B, <16 x i8> %C, <8 x i8> %D) {
; CHECK: tbx2_8b
; CHECK: tbx.8b
  %tmp3 = call <8 x i8> @llvm.aarch64.neon.tbx2.v8i8(<8 x i8> %A, <16 x i8> %B, <16 x i8> %C, <8 x i8> %D)
  ret <8 x i8> %tmp3
}

define <16 x i8> @tbx2_16b(<16 x i8> %A, <16 x i8> %B, <16 x i8> %C, <16 x i8> %D) {
; CHECK: tbx2_16b
; CHECK: tbx.16b
  %tmp3 = call <16 x i8> @llvm.aarch64.neon.tbx2.v16i8(<16 x i8> %A, <16 x i8> %B, <16 x i8> %C, <16 x i8> %D)
  ret <16 x i8> %tmp3
}

define <8 x i8> @tbx3_8b(<8 x i8> %A, <16 x i8> %B, <16 x i8> %C, <16 x i8> %D, <8 x i8> %E) {
; CHECK: tbx3_8b
; CHECK: tbx.8b
  %tmp3 = call <8 x i8> @llvm.aarch64.neon.tbx3.v8i8(< 8 x i8> %A, <16 x i8> %B, <16 x i8> %C, <16 x i8> %D, <8 x i8> %E)
  ret <8 x i8> %tmp3
}

define <16 x i8> @tbx3_16b(<16 x i8> %A, <16 x i8> %B, <16 x i8> %C, <16 x i8> %D, <16 x i8> %E) {
; CHECK: tbx3_16b
; CHECK: tbx.16b
  %tmp3 = call <16 x i8> @llvm.aarch64.neon.tbx3.v16i8(<16 x i8> %A, <16 x i8> %B, <16 x i8> %C, <16 x i8> %D, <16 x i8> %E)
  ret <16 x i8> %tmp3
}

define <8 x i8> @tbx4_8b(<8 x i8> %A, <16 x i8> %B, <16 x i8> %C, <16 x i8> %D, <16 x i8> %E, <8 x i8> %F) {
; CHECK: tbx4_8b
; CHECK: tbx.8b
  %tmp3 = call <8 x i8> @llvm.aarch64.neon.tbx4.v8i8(<8 x i8> %A, <16 x i8> %B, <16 x i8> %C, <16 x i8> %D, <16 x i8> %E, <8 x i8> %F)
  ret <8 x i8> %tmp3
}

define <16 x i8> @tbx4_16b(<16 x i8> %A, <16 x i8> %B, <16 x i8> %C, <16 x i8> %D, <16 x i8> %E, <16 x i8> %F) {
; CHECK: tbx4_16b
; CHECK: tbx.16b
  %tmp3 = call <16 x i8> @llvm.aarch64.neon.tbx4.v16i8(<16 x i8> %A, <16 x i8> %B, <16 x i8> %C, <16 x i8> %D, <16 x i8> %E, <16 x i8> %F)
  ret <16 x i8> %tmp3
}

declare <8 x i8> @llvm.aarch64.neon.tbx1.v8i8(<8 x i8>, <16 x i8>, <8 x i8>) nounwind readnone
declare <16 x i8> @llvm.aarch64.neon.tbx1.v16i8(<16 x i8>, <16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i8> @llvm.aarch64.neon.tbx2.v8i8(<8 x i8>, <16 x i8>, <16 x i8>, <8 x i8>) nounwind readnone
declare <16 x i8> @llvm.aarch64.neon.tbx2.v16i8(<16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i8> @llvm.aarch64.neon.tbx3.v8i8(<8 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <8 x i8>) nounwind readnone
declare <16 x i8> @llvm.aarch64.neon.tbx3.v16i8(<16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i8> @llvm.aarch64.neon.tbx4.v8i8(<8 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <8 x i8>) nounwind readnone
declare <16 x i8> @llvm.aarch64.neon.tbx4.v16i8(<16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>) nounwind readnone

