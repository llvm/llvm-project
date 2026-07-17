; RUN: opt -passes="print<cost-model>" -disable-output < %s
; This test triggers a crash in X86 TTI with scalable vectors

target triple = "x86_64-unknown-linux-gnu"

define <vscale x 1 x i64> @test(i64 %x) {
entry:
  %v = insertelement <vscale x 1 x i64> poison, i64 %x, i64 0
  ret <vscale x 1 x i64> %v
}
