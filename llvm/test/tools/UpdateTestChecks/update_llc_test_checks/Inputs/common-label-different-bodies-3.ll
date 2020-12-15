; RUN: llc < %s -mtriple=i686-unknown-linux-gnu -mattr=+sse2 | FileCheck %s --check-prefixes=A,B
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu | FileCheck %s --check-prefixes=A,C

declare <2 x i64> @llvm.bswap.v2i64(<2 x i64>)
; A: declare <2 x i64> @llvm.bswap.v2i64(<2 x i64>)

define <2 x i64> @fold_v2i64() {
entry:
  %r = call <2 x i64> @llvm.bswap.v2i64(<2 x i64> <i64 255, i64 -1>)
  ret <2 x i64> %r
}
