; RUN: llc -verify-machineinstrs -mcpu=pwr9 -mtriple=powerpc64-unknown-linux-gnu  < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mcpu=pwr9 -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s

; Check the vabsd* instructions that were added in PowerISA V3.0

; Function Attrs: nounwind readnone
declare <16 x i8> @llvm.ppc.altivec.vabsdub(<16 x i8>, <16 x i8>)

; Function Attrs: nounwind readnone
declare <8 x i16> @llvm.ppc.altivec.vabsduh(<8 x i16>, <8 x i16>)

; Function Attrs: nounwind readnone
declare <4 x i32> @llvm.ppc.altivec.vabsduw(<4 x i32>, <4 x i32>)

define <16 x i8> @test_byte(<16 x i8> %a, <16 x i8> %b) {
entry:
  %res = tail call <16 x i8> @llvm.ppc.altivec.vabsdub(<16 x i8> %a, <16 x i8> %b)
  ret <16 x i8> %res
; CHECK-LABEL: @test_byte
; CHECK: vabsdub 2, 2, 3
; CHECK: blr
}

define <8 x i16> @test_half(<8 x i16> %a, <8 x i16> %b) {
entry:
  %res = tail call <8 x i16> @llvm.ppc.altivec.vabsduh(<8 x i16> %a, <8 x i16> %b)
  ret <8 x i16> %res
; CHECK-LABEL: @test_half
; CHECK: vabsduh 2, 2, 3
; CHECK: blr
}

define <4 x i32> @test_word(<4 x i32> %a, <4 x i32> %b) {
entry:
  %res = tail call <4 x i32> @llvm.ppc.altivec.vabsduw(<4 x i32> %a, <4 x i32> %b)
  ret <4 x i32> %res
; CHECK-LABEL: @test_word
; CHECK: vabsduw 2, 2, 3
; CHECK: blr
}
