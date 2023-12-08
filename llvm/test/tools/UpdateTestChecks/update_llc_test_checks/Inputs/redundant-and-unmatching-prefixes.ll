; RUN: llc < %s -mtriple=i686-unknown-linux-gnu -mattr=+sse2 | FileCheck %s --check-prefixes=A,B,REDUNDANT
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu | FileCheck %s --check-prefixes=C,A,UNUSED

; prefix 'A' has conflicting outputs, while the REDUNDANT and UNUSED ones are
; unused
declare <2 x i64> @llvm.bswap.v2i64(<2 x i64>)

define <2 x i64> @function_1() {
entry:
  %r = call <2 x i64> @llvm.bswap.v2i64(<2 x i64> <i64 255, i64 -1>)
  ret <2 x i64> %r
}

define <2 x i64> @function_2() {
entry:
  %r = call <2 x i64> @llvm.bswap.v2i64(<2 x i64> <i64 16, i64 -1>)
  ret <2 x i64> %r
}
