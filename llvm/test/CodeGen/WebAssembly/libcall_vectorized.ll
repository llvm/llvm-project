
; RUN: llc < %s -disable-wasm-fallthrough-return-opt -wasm-keep-registers  -mattr=+simd128 | FileCheck %s

target triple = "wasm32-unknown-unknown"

declare <4 x float> @llvm.exp10.v4f32(<4 x float>)

; XFAIL: target={{.*}}
define <4 x float> @exp10_f32v4(<4 x float> %v) {
entry:
  %r = call <4 x float> @llvm.exp10.v4f32(<4 x float> %v)
  ret <4 x float> %r
}
