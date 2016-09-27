; RUN: llc -mcpu=pwr9 -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -verify-machineinstrs < %s | FileCheck %s

; Function Attrs: nounwind readnone
define zeroext i32 @testCTZ32(i32 signext %a) {
entry:
  %0 = tail call i32 @llvm.cttz.i32(i32 %a, i1 false)
  ret i32 %0
; CHECK-LABEL: testCTZ32
; CHECK: cnttzw 3, 3
}

; Function Attrs: nounwind readnone
declare i32 @llvm.cttz.i32(i32, i1)

; Function Attrs: nounwind readnone
define zeroext i32 @testCTZ64(i64 %a) {
entry:
  %0 = tail call i64 @llvm.cttz.i64(i64 %a, i1 false)
  %cast = trunc i64 %0 to i32
  ret i32 %cast
; CHECK-LABEL: testCTZ64
; CHECK: cnttzd 3, 3
}

; Function Attrs: nounwind readnone
declare i64 @llvm.cttz.i64(i64, i1)

; Function Attrs: nounwind readnone
define <16 x i8> @testVCMPNEB(<16 x i8> %a, <16 x i8> %b) {
entry:
  %0 = tail call <16 x i8> @llvm.ppc.altivec.vcmpneb(<16 x i8> %a, <16 x i8> %b)
  ret <16 x i8> %0
; CHECK-LABEL: testVCMPNEB
; CHECK: vcmpneb 2, 2
}

; Function Attrs: nounwind readnone
declare <16 x i8> @llvm.ppc.altivec.vcmpneb(<16 x i8>, <16 x i8>)

; Function Attrs: nounwind readnone
define <16 x i8> @testVCMPNEZB(<16 x i8> %a, <16 x i8> %b) {
entry:
  %0 = tail call <16 x i8> @llvm.ppc.altivec.vcmpnezb(<16 x i8> %a, <16 x i8> %b)
  ret <16 x i8> %0
; CHECK-LABEL: testVCMPNEZB
; CHECK: vcmpnezb 2, 2
}

; Function Attrs: nounwind readnone
declare <16 x i8> @llvm.ppc.altivec.vcmpnezb(<16 x i8>, <16 x i8>)

; Function Attrs: nounwind readnone
define <8 x i16> @testVCMPNEH(<8 x i16> %a, <8 x i16> %b) {
entry:
  %0 = tail call <8 x i16> @llvm.ppc.altivec.vcmpneh(<8 x i16> %a, <8 x i16> %b)
  ret <8 x i16> %0
; CHECK-LABEL: testVCMPNEH
; CHECK: vcmpneh 2, 2
}

; Function Attrs: nounwind readnone
declare <8 x i16> @llvm.ppc.altivec.vcmpneh(<8 x i16>, <8 x i16>)

; Function Attrs: nounwind readnone
define <8 x i16> @testVCMPNEZH(<8 x i16> %a, <8 x i16> %b) {
entry:
  %0 = tail call <8 x i16> @llvm.ppc.altivec.vcmpnezh(<8 x i16> %a, <8 x i16> %b)
  ret <8 x i16> %0
; CHECK-LABEL: testVCMPNEZH
; CHECK: vcmpnezh 2, 2
}

; Function Attrs: nounwind readnone
declare <8 x i16> @llvm.ppc.altivec.vcmpnezh(<8 x i16>, <8 x i16>)

; Function Attrs: nounwind readnone
define <4 x i32> @testVCMPNEW(<4 x i32> %a, <4 x i32> %b) {
entry:
  %0 = tail call <4 x i32> @llvm.ppc.altivec.vcmpnew(<4 x i32> %a, <4 x i32> %b)
  ret <4 x i32> %0
; CHECK-LABEL: testVCMPNEW
; CHECK: vcmpnew 2, 2
}

; Function Attrs: nounwind readnone
declare <4 x i32> @llvm.ppc.altivec.vcmpnew(<4 x i32>, <4 x i32>)

; Function Attrs: nounwind readnone
define <4 x i32> @testVCMPNEZW(<4 x i32> %a, <4 x i32> %b) {
entry:
  %0 = tail call <4 x i32> @llvm.ppc.altivec.vcmpnezw(<4 x i32> %a, <4 x i32> %b)
  ret <4 x i32> %0
; CHECK-LABEL: testVCMPNEZW
; CHECK: vcmpnezw 2, 2
}

; Function Attrs: nounwind readnone
declare <4 x i32> @llvm.ppc.altivec.vcmpnezw(<4 x i32>, <4 x i32>)

; Function Attrs: nounwind readnone
define <16 x i8> @testVCNTTZB(<16 x i8> %a) {
entry:
  %0 = tail call <16 x i8> @llvm.cttz.v16i8(<16 x i8> %a, i1 false)
  ret <16 x i8> %0
; CHECK-LABEL: testVCNTTZB
; CHECK: vctzb 2, 2
}

; Function Attrs: nounwind readnone
define <8 x i16> @testVCNTTZH(<8 x i16> %a) {
entry:
  %0 = tail call <8 x i16> @llvm.cttz.v8i16(<8 x i16> %a, i1 false)
  ret <8 x i16> %0
; CHECK-LABEL: testVCNTTZH
; CHECK: vctzh 2, 2
}

; Function Attrs: nounwind readnone
define <4 x i32> @testVCNTTZW(<4 x i32> %a) {
entry:
  %0 = tail call <4 x i32> @llvm.cttz.v4i32(<4 x i32> %a, i1 false)
  ret <4 x i32> %0
; CHECK-LABEL: testVCNTTZW
; CHECK: vctzw 2, 2
}

; Function Attrs: nounwind readnone
define <2 x i64> @testVCNTTZD(<2 x i64> %a) {
entry:
  %0 = tail call <2 x i64> @llvm.cttz.v2i64(<2 x i64> %a, i1 false)
  ret <2 x i64> %0
; CHECK-LABEL: testVCNTTZD
; CHECK: vctzd 2, 2
}

; Function Attrs: nounwind readnone
declare <16 x i8> @llvm.cttz.v16i8(<16 x i8>, i1)

; Function Attrs: nounwind readnone
declare <8 x i16> @llvm.cttz.v8i16(<8 x i16>, i1)

; Function Attrs: nounwind readnone
declare <4 x i32> @llvm.cttz.v4i32(<4 x i32>, i1)

; Function Attrs: nounwind readnone
declare <2 x i64> @llvm.cttz.v2i64(<2 x i64>, i1)
