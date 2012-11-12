; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=core-avx2 -mattr=+avx2 | FileCheck %s

; CHECK: vpcmpgtd  %ymm
define <8 x i32> @int256-cmp(<8 x i32> %i, <8 x i32> %j) nounwind readnone {
  %bincmp = icmp slt <8 x i32> %i, %j
  %x = sext <8 x i1> %bincmp to <8 x i32>
  ret <8 x i32> %x
}

; CHECK: vpcmpgtq  %ymm
define <4 x i64> @v4i64-cmp(<4 x i64> %i, <4 x i64> %j) nounwind readnone {
  %bincmp = icmp slt <4 x i64> %i, %j
  %x = sext <4 x i1> %bincmp to <4 x i64>
  ret <4 x i64> %x
}

; CHECK: vpcmpgtw  %ymm
define <16 x i16> @v16i16-cmp(<16 x i16> %i, <16 x i16> %j) nounwind readnone {
  %bincmp = icmp slt <16 x i16> %i, %j
  %x = sext <16 x i1> %bincmp to <16 x i16>
  ret <16 x i16> %x
}

; CHECK: vpcmpgtb  %ymm
define <32 x i8> @v32i8-cmp(<32 x i8> %i, <32 x i8> %j) nounwind readnone {
  %bincmp = icmp slt <32 x i8> %i, %j
  %x = sext <32 x i1> %bincmp to <32 x i8>
  ret <32 x i8> %x
}

; CHECK: vpcmpeqd  %ymm
define <8 x i32> @int256-cmpeq(<8 x i32> %i, <8 x i32> %j) nounwind readnone {
  %bincmp = icmp eq <8 x i32> %i, %j
  %x = sext <8 x i1> %bincmp to <8 x i32>
  ret <8 x i32> %x
}

; CHECK: vpcmpeqq  %ymm
define <4 x i64> @v4i64-cmpeq(<4 x i64> %i, <4 x i64> %j) nounwind readnone {
  %bincmp = icmp eq <4 x i64> %i, %j
  %x = sext <4 x i1> %bincmp to <4 x i64>
  ret <4 x i64> %x
}

; CHECK: vpcmpeqw  %ymm
define <16 x i16> @v16i16-cmpeq(<16 x i16> %i, <16 x i16> %j) nounwind readnone {
  %bincmp = icmp eq <16 x i16> %i, %j
  %x = sext <16 x i1> %bincmp to <16 x i16>
  ret <16 x i16> %x
}

; CHECK: vpcmpeqb  %ymm
define <32 x i8> @v32i8-cmpeq(<32 x i8> %i, <32 x i8> %j) nounwind readnone {
  %bincmp = icmp eq <32 x i8> %i, %j
  %x = sext <32 x i1> %bincmp to <32 x i8>
  ret <32 x i8> %x
}

