; RUN: llc -O2 -mtriple=aarch64_be-linux-gnu -verify-machineinstrs < %s | FileCheck %s --check-prefix=BE

; Regression test for https://github.com/llvm/llvm-project/issues/221122.
; A promoted byte extract from an all-ones boolean mask must be sign-extended
; before a wider-lane DUP consumes it.

define i1 @check(<16 x i32> %fr, <4 x i32> %fr1) {
; BE-LABEL: check:
; BE:       smov w[[MASK:[0-9]+]], v[[SRC:[0-9]+]].b[0]
; BE:       dup v[[DUP:[0-9]+]].4h, w[[MASK]]
; BE:       and {{.*}}v[[DUP]].8b
; BE:       uminv b0,
entry:
  %cmp16 = icmp eq <16 x i32> %fr, <i32 3208, i32 1334, i32 28764, i32 35679, i32 2789, i32 13028, i32 4754, i32 168364, i32 91254, i32 12399, i32 22848, i32 8174, i32 307964, i32 146829, i32 22009, i32 32668>
  %cmp4 = icmp eq <4 x i32> %fr1, <i32 11594, i32 447564, i32 202404, i32 31619>
  %splat = shufflevector <16 x i1> %cmp16, <16 x i1> poison, <4 x i32> zeroinitializer
  %both = and <4 x i1> %splat, %cmp4
  %pad = shufflevector <4 x i1> %both, <4 x i1> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4>
  %merge = shufflevector <16 x i1> %pad, <16 x i1> %cmp16, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %bits = bitcast <16 x i1> %merge to i16
  %ok = icmp eq i16 %bits, -1
  ret i1 %ok
}
