; RUN: llc < %s -mtriple=aarch64-eabi -aarch64-neon-syntax=generic | FileCheck %s

define i8 @add_B(<16 x i8>* %arr)  {
; CHECK-LABEL: add_B
; CHECK: addv {{b[0-9]+}}, {{v[0-9]+}}.16b
  %bin.rdx = load <16 x i8>, <16 x i8>* %arr
  %rdx.shuf0 = shufflevector <16 x i8> %bin.rdx, <16 x i8> undef, <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 undef, i32 undef,i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx0 = add <16 x i8> %bin.rdx, %rdx.shuf0
  %rdx.shuf = shufflevector <16 x i8> %bin.rdx0, <16 x i8> undef, <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef,i32 undef, i32 undef, i32 undef, i32 undef,i32 undef, i32 undef, i32 undef, i32 undef,i32 undef, i32 undef >
  %bin.rdx11 = add <16 x i8> %bin.rdx0, %rdx.shuf
  %rdx.shuf12 = shufflevector <16 x i8> %bin.rdx11, <16 x i8> undef, <16 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef,i32 undef, i32 undef, i32 undef, i32 undef,i32 undef, i32 undef>
  %bin.rdx13 = add <16 x i8> %bin.rdx11, %rdx.shuf12
  %rdx.shuf13 = shufflevector <16 x i8> %bin.rdx13, <16 x i8> undef, <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef,i32 undef, i32 undef, i32 undef, i32 undef,i32 undef, i32 undef>
  %bin.rdx14 = add <16 x i8> %bin.rdx13, %rdx.shuf13
  %r = extractelement <16 x i8> %bin.rdx14, i32 0
  ret i8 %r
}

define i16 @add_H(<8 x i16>* %arr)  {
; CHECK-LABEL: add_H
; CHECK: addv {{h[0-9]+}}, {{v[0-9]+}}.8h
  %bin.rdx = load <8 x i16>, <8 x i16>* %arr
  %rdx.shuf = shufflevector <8 x i16> %bin.rdx, <8 x i16> undef, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef,i32 undef, i32 undef>
  %bin.rdx11 = add <8 x i16> %bin.rdx, %rdx.shuf
  %rdx.shuf12 = shufflevector <8 x i16> %bin.rdx11, <8 x i16> undef, <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx13 = add <8 x i16> %bin.rdx11, %rdx.shuf12
  %rdx.shuf13 = shufflevector <8 x i16> %bin.rdx13, <8 x i16> undef, <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx14 = add <8 x i16> %bin.rdx13, %rdx.shuf13
  %r = extractelement <8 x i16> %bin.rdx14, i32 0
  ret i16 %r
}

define i32 @add_S( <4 x i32>* %arr)  {
; CHECK-LABEL: add_S
; CHECK: addv {{s[0-9]+}}, {{v[0-9]+}}.4s
  %bin.rdx = load <4 x i32>, <4 x i32>* %arr
  %rdx.shuf = shufflevector <4 x i32> %bin.rdx, <4 x i32> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %bin.rdx11 = add <4 x i32> %bin.rdx, %rdx.shuf
  %rdx.shuf12 = shufflevector <4 x i32> %bin.rdx11, <4 x i32> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %bin.rdx13 = add <4 x i32> %bin.rdx11, %rdx.shuf12
  %r = extractelement <4 x i32> %bin.rdx13, i32 0
  ret i32 %r
}

define i64 @add_D(<2 x i64>* %arr)  {
; CHECK-LABEL: add_D
; CHECK-NOT: addv
  %bin.rdx = load <2 x i64>, <2 x i64>* %arr
  %rdx.shuf0 = shufflevector <2 x i64> %bin.rdx, <2 x i64> undef, <2 x i32> <i32 1, i32 undef>
  %bin.rdx0 = add <2 x i64> %bin.rdx, %rdx.shuf0
  %r = extractelement <2 x i64> %bin.rdx0, i32 0
  ret i64 %r
}

define i32 @oversized_ADDV_256(i8* noalias nocapture readonly %arg1, i8* noalias nocapture readonly %arg2) {
; CHECK-LABEL: oversized_ADDV_256
; CHECK: addv {{s[0-9]+}}, {{v[0-9]+}}.4s
entry:
  %0 = bitcast i8* %arg1 to <8 x i8>*
  %1 = load <8 x i8>, <8 x i8>* %0, align 1
  %2 = zext <8 x i8> %1 to <8 x i32>
  %3 = bitcast i8* %arg2 to <8 x i8>*
  %4 = load <8 x i8>, <8 x i8>* %3, align 1
  %5 = zext <8 x i8> %4 to <8 x i32>
  %6 = sub nsw <8 x i32> %2, %5
  %7 = icmp slt <8 x i32> %6, zeroinitializer
  %8 = sub nsw <8 x i32> zeroinitializer, %6
  %9 = select <8 x i1> %7, <8 x i32> %8, <8 x i32> %6
  %rdx.shuf = shufflevector <8 x i32> %9, <8 x i32> undef, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx = add <8 x i32> %9, %rdx.shuf
  %rdx.shuf1 = shufflevector <8 x i32> %bin.rdx, <8 x i32> undef, <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx2 = add <8 x i32> %bin.rdx, %rdx.shuf1
  %rdx.shuf3 = shufflevector <8 x i32> %bin.rdx2, <8 x i32> undef, <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx4 = add <8 x i32> %bin.rdx2, %rdx.shuf3
  %10 = extractelement <8 x i32> %bin.rdx4, i32 0
  ret i32 %10
}

define i32 @oversized_ADDV_512(<16 x i32>* %arr)  {
; CHECK-LABEL: oversized_ADDV_512
; CHECK: addv {{s[0-9]+}}, {{v[0-9]+}}.4s
  %bin.rdx = load <16 x i32>, <16 x i32>* %arr

  %rdx.shuf0 = shufflevector <16 x i32> %bin.rdx, <16 x i32> undef, <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 undef, i32 undef,i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx0 = add <16 x i32> %bin.rdx, %rdx.shuf0

  %rdx.shuf = shufflevector <16 x i32> %bin.rdx0, <16 x i32> undef, <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef,i32 undef, i32 undef, i32 undef, i32 undef,i32 undef, i32 undef, i32 undef, i32 undef,i32 undef, i32 undef >
  %bin.rdx11 = add <16 x i32> %bin.rdx0, %rdx.shuf

  %rdx.shuf12 = shufflevector <16 x i32> %bin.rdx11, <16 x i32> undef, <16 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef,i32 undef, i32 undef, i32 undef, i32 undef,i32 undef, i32 undef>
  %bin.rdx13 = add <16 x i32> %bin.rdx11, %rdx.shuf12

  %rdx.shuf13 = shufflevector <16 x i32> %bin.rdx13, <16 x i32> undef, <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef,i32 undef, i32 undef, i32 undef, i32 undef,i32 undef, i32 undef>
  %bin.rdx14 = add <16 x i32> %bin.rdx13, %rdx.shuf13

  %r = extractelement <16 x i32> %bin.rdx14, i32 0
  ret i32 %r
}
