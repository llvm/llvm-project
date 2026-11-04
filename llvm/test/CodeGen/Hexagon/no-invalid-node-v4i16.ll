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
; CHECK-LABEL: sq77777777:
; CHECK: r{{[0-9]+}}:{{[0-9]+}} = vasrh(r{{[0-9]+}}:{{[0-9]+}},#1)

target triple = "hexagon-unknown-linux-musl"

define <8 x i16> @sq77777777(<8 x i16> %0) {
entry:
  %div = sdiv <8 x i16> %0, splat (i16 7)
  ret <8 x i16> %div
}
