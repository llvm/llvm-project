; RUN: llc < %s -mtriple=i686-unknown-linux-gnu -mattr=+avx2 -verify-machineinstrs -o /dev/null
; RUN: llc < %s -mtriple=i686-unknown-linux-gnu -mattr=+avx512f,-avx512dq -verify-machineinstrs -o /dev/null
; RUN: llc < %s -mtriple=i686-unknown-linux-gnu -mattr=+avx512f,+avx512dq,+avx512vl -verify-machineinstrs -o /dev/null

; Constant signed division/remainder must not unroll vector MUL_LOHI into
; scalar i64 MUL_LOHI on 32-bit targets, where i64 is an illegal type.

define <4 x i64> @sdiv_v4i64(<4 x i64> %a) {
  %r = sdiv <4 x i64> %a, splat (i64 62)
  ret <4 x i64> %r
}

define <4 x i64> @srem_v4i64(<4 x i64> %a) {
  %r = srem <4 x i64> %a, splat (i64 62)
  ret <4 x i64> %r
}

define <8 x i64> @sdiv_v8i64(<8 x i64> %a) {
  %r = sdiv <8 x i64> %a, splat (i64 62)
  ret <8 x i64> %r
}

define <8 x i64> @srem_v8i64(<8 x i64> %a) {
  %r = srem <8 x i64> %a, splat (i64 62)
  ret <8 x i64> %r
}

define <4 x i64> @sdiv_v4i64_negative(<4 x i64> %a) {
  %r = sdiv <4 x i64> %a, splat (i64 -62)
  ret <4 x i64> %r
}

define <4 x i64> @srem_v4i64_negative(<4 x i64> %a) {
  %r = srem <4 x i64> %a, splat (i64 -62)
  ret <4 x i64> %r
}

define <8 x i64> @sdiv_v8i64_negative(<8 x i64> %a) {
  %r = sdiv <8 x i64> %a, splat (i64 -62)
  ret <8 x i64> %r
}

define <8 x i64> @srem_v8i64_negative(<8 x i64> %a) {
  %r = srem <8 x i64> %a, splat (i64 -62)
  ret <8 x i64> %r
}
