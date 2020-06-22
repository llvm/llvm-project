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

define <16 x i8> @test_vabsdub(<16 x i8> %0, <16 x i8> %1) {
entry:
  %2 = zext <16 x i8> %0 to <16 x i32>
  %3 = zext <16 x i8> %1 to <16 x i32>
  %4 = sub nsw <16 x i32> %2, %3
  %5 = icmp slt <16 x i32> %4, zeroinitializer
  %6 = sub nsw <16 x i32> zeroinitializer, %4
  %7 = select <16 x i1> %5, <16 x i32> %6, <16 x i32> %4
  %8 = trunc <16 x i32> %7 to <16 x i8>
  ret <16 x i8> %8
; CHECK-LABEL: @test_vabsdub
; CHECK: vabsdub 2, 2, 3
; CHECK: blr
}

define <8 x i16> @test_vabsduh(<8 x i16> %0, <8 x i16> %1) {
entry:
  %2 = zext <8 x i16> %0 to <8 x i32>
  %3 = zext <8 x i16> %1 to <8 x i32>
  %4 = sub nsw <8 x i32> %2, %3
  %5 = icmp slt <8 x i32> %4, zeroinitializer
  %6 = sub nsw <8 x i32> zeroinitializer, %4
  %7 = select <8 x i1> %5, <8 x i32> %6, <8 x i32> %4
  %8 = trunc <8 x i32> %7 to <8 x i16>
  ret <8 x i16> %8
; CHECK-LABEL: @test_vabsduh
; CHECK: vabsduh 2, 2, 3
; CHECK: blr
}

define <4 x i32> @test_vabsduw(<4 x i32> %0, <4 x i32> %1) {
entry:
  %2 = sub nsw <4 x i32> %0, %1
  %3 = icmp slt <4 x i32> %2, zeroinitializer
  %4 = sub nsw <4 x i32> zeroinitializer, %2
  %5 = select <4 x i1> %3, <4 x i32> %4, <4 x i32> %2
  ret <4 x i32> %5
; CHECK-LABEL: @test_vabsduw
; CHECK: vabsduw 2, 2, 3
; CHECK: blr
}
