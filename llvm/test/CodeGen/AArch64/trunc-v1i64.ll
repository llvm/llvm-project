; RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+neon -verify-machineinstrs < %s | FileCheck %s

; An optimization in DAG Combiner to fold
; (trunc (concat ... x ...)) -> (concat ..., (trunc x), ...))
; will generate nodes like:
;     v1i32 trunc v1i64, v1i16 trunc v1i64, v1i8 trunc v1i64.
; And such nodes will be defaultly scalarized in type legalization. But such
; scalarization will cause an assertion failure, as v1i64 is a legal type in
; AArch64. We change the default behaviour from be scalarized to be widen.

; FIXME: Currently XTN is generated for v1i32, but it can be optimized.
; Just like v1i16 and v1i8, there is no XTN generated.

define <2 x i32> @test_v1i32_0(<1 x i64> %in0) {
; CHECK-LABEL: test_v1i32_0:
; CHECK: xtn v0.2s, v0.2d
  %1 = shufflevector <1 x i64> %in0, <1 x i64> undef, <2 x i32> <i32 0, i32 undef>
  %2 = trunc <2 x i64> %1 to <2 x i32>
  ret <2 x i32> %2
}

define <2 x i32> @test_v1i32_1(<1 x i64> %in0) {
; CHECK-LABEL: test_v1i32_1:
; CHECK: xtn v0.2s, v0.2d
; CHECK-NEXT: dup v0.2s, v0.s[0]
  %1 = shufflevector <1 x i64> %in0, <1 x i64> undef, <2 x i32> <i32 undef, i32 0>
  %2 = trunc <2 x i64> %1 to <2 x i32>
  ret <2 x i32> %2
}

define <4 x i16> @test_v1i16_0(<1 x i64> %in0) {
; CHECK-LABEL: test_v1i16_0:
; CHECK-NOT: xtn
  %1 = shufflevector <1 x i64> %in0, <1 x i64> undef, <4 x i32> <i32 0, i32 undef, i32 undef, i32 undef>
  %2 = trunc <4 x i64> %1 to <4 x i16>
  ret <4 x i16> %2
}

define <4 x i16> @test_v1i16_1(<1 x i64> %in0) {
; CHECK-LABEL: test_v1i16_1:
; CHECK-NOT: xtn
; CHECK: dup v0.4h, v0.h[0]
  %1 = shufflevector <1 x i64> %in0, <1 x i64> undef, <4 x i32> <i32 undef, i32 undef, i32 0, i32 undef>
  %2 = trunc <4 x i64> %1 to <4 x i16>
  ret <4 x i16> %2
}

define <8 x i8> @test_v1i8_0(<1 x i64> %in0) {
; CHECK-LABEL: test_v1i8_0:
; CHECK-NOT: xtn
  %1 = shufflevector <1 x i64> %in0, <1 x i64> undef, <8 x i32> <i32 0, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %2 = trunc <8 x i64> %1 to <8 x i8>
  ret <8 x i8> %2
}

define <8 x i8> @test_v1i8_1(<1 x i64> %in0) {
; CHECK-LABEL: test_v1i8_1:
; CHECK-NOT: xtn
; CHECK: dup v0.8b, v0.b[0]
  %1 = shufflevector <1 x i64> %in0, <1 x i64> undef, <8 x i32> <i32 undef, i32 undef, i32 0, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %2 = trunc <8 x i64> %1 to <8 x i8>
  ret <8 x i8> %2
}

; PR20777: v1i1 is also problematic, but we can't widen it, so we extract_elt
; the i64 out of the v1i64 operand, and truncate that scalar instead.

define <1 x i1> @test_v1i1_0(<1 x i64> %in0) {
; CHECK-LABEL: test_v1i1_0:
; CHECK: fmov w0, s0
  %1 = trunc <1 x i64> %in0 to <1 x i1>
  ret <1 x i1> %1
}

define i1 @test_v1i1_1(<1 x i64> %in0) {
; CHECK-LABEL: test_v1i1_1:
; CHECK: fmov [[REG:w[0-9]+]], s0
  %1 = trunc <1 x i64> %in0 to <1 x i1>
; CHECK: and w0, [[REG]], #0x1
  %2 = extractelement <1 x i1> %1, i32 0
  ret i1 %2
}
