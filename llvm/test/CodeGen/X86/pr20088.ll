; RUN: llc < %s -march=x86-64 -mattr=+avx | FileCheck %s

declare <16 x i8> @llvm.x86.sse41.pblendvb(<16 x i8>, <16 x i8>, <16 x i8>)

define <16 x i8> @foo(<16 x i8> %x) {
; CHECK: vpblendvb
  %res = call <16 x i8> @llvm.x86.sse41.pblendvb(<16 x i8> zeroinitializer, <16 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>, <16 x i8> %x)
  ret <16 x i8> %res;
}
