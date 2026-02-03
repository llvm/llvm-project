; REQUIRES: asserts
; RUN: llc -march=hexagon -verify-machineinstrs -o - < %s 2>&1 | FileCheck %s

; This is a crash / fatal-error regression test:
; llc used to hit:
;   LLVM ERROR: invalid node: operand #1 must have type i32, but has type i16
; during DAG combine / ISel:
;   t61: v4i16 = HexagonISD::VASR t56, Constant:i16<1>
;   t56: v4i16 = mulhs ...
;
; The test ensures llc does NOT emit "LLVM ERROR" and produces assembly for the function.

; CHECK-NOT: LLVM ERROR:
; CHECK-NOT: invalid node:

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon-unknown-linux-musl"

define <8 x i16> @sq77777777(<8 x i16> %0) {
; CHECK-LABEL: sq77777777:
entry:
  %div = sdiv <8 x i16> %0, splat (i16 7)
  ret <8 x i16> %div
}
