; RUN: llc < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnu"

; This test verifies that we can enable subtarget features via
; the function attributes and generate appropriate code (or,
; in this case, select the instruction at all).

; Function Attrs: nounwind
define <16 x i8> @foo(<16 x i8> %data, <16 x i8> %key) #0 {
  %vaeseq_v.i = call <16 x i8> @llvm.aarch64.crypto.aese(<16 x i8> %data, <16 x i8> %key)
  ret <16 x i8> %vaeseq_v.i
}

; CHECK: foo
; CHECK: aese

; Function Attrs: nounwind readnone
declare <16 x i8> @llvm.aarch64.crypto.aese(<16 x i8>, <16 x i8>)

attributes #0 = { nounwind "target-features"="+neon,+crc,+crypto" }
