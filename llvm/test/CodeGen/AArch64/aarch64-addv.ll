; RUN: llc < %s -mtriple=aarch64-eabi -aarch64-neon-syntax=generic | FileCheck %s

; Function Attrs: nounwind readnone
declare i64 @llvm.experimental.vector.reduce.add.v2i64(<2 x i64>)
declare i32 @llvm.experimental.vector.reduce.add.v4i32(<4 x i32>)
declare i16 @llvm.experimental.vector.reduce.add.v8i16(<8 x i16>)
declare i8 @llvm.experimental.vector.reduce.add.v16i8(<16 x i8>)

define i8 @add_B(<16 x i8>* %arr)  {
; CHECK-LABEL: add_B
; CHECK: addv {{b[0-9]+}}, {{v[0-9]+}}.16b
  %bin.rdx = load <16 x i8>, <16 x i8>* %arr
  %r = call i8 @llvm.experimental.vector.reduce.add.v16i8(<16 x i8> %bin.rdx)
  ret i8 %r
}

define i16 @add_H(<8 x i16>* %arr)  {
; CHECK-LABEL: add_H
; CHECK: addv {{h[0-9]+}}, {{v[0-9]+}}.8h
  %bin.rdx = load <8 x i16>, <8 x i16>* %arr
  %r = call i16 @llvm.experimental.vector.reduce.add.v8i16(<8 x i16> %bin.rdx)
  ret i16 %r
}

define i32 @add_S( <4 x i32>* %arr)  {
; CHECK-LABEL: add_S
; CHECK: addv {{s[0-9]+}}, {{v[0-9]+}}.4s
  %bin.rdx = load <4 x i32>, <4 x i32>* %arr
  %r = call i32 @llvm.experimental.vector.reduce.add.v4i32(<4 x i32> %bin.rdx)
  ret i32 %r
}

define i64 @add_D(<2 x i64>* %arr)  {
; CHECK-LABEL: add_D
; CHECK-NOT: addv
  %bin.rdx = load <2 x i64>, <2 x i64>* %arr
  %r = call i64 @llvm.experimental.vector.reduce.add.v2i64(<2 x i64> %bin.rdx)
  ret i64 %r
}

declare i32 @llvm.experimental.vector.reduce.add.v8i32(<8 x i32>)

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
  %r = call i32 @llvm.experimental.vector.reduce.add.v8i32(<8 x i32> %9)
  ret i32 %r
}

declare i32 @llvm.experimental.vector.reduce.add.v16i32(<16 x i32>)

define i32 @oversized_ADDV_512(<16 x i32>* %arr)  {
; CHECK-LABEL: oversized_ADDV_512
; CHECK: addv {{s[0-9]+}}, {{v[0-9]+}}.4s
  %bin.rdx = load <16 x i32>, <16 x i32>* %arr
  %r = call i32 @llvm.experimental.vector.reduce.add.v16i32(<16 x i32> %bin.rdx)
  ret i32 %r
}
