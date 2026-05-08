; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; Check that non-uniform udiv of v8i16 compiles correctly.
; The udiv expansion produces magic-number constants (e.g. 0xAAAB) that
; don't fit in signed 16-bit, which previously triggered an APInt assertion
; in getBuildVectorConstInts.

; CHECK-LABEL: uq65656565:
; CHECK-DAG:   vmpyh(r{{[0-9]+}},r{{[0-9]+}}):sat
; CHECK-DAG:   vlsrh(r{{[0-9]+}}:{{[0-9]+}},#2)
; CHECK:       jumpr r31
define <8 x i16> @uq65656565(<8 x i16> %0) {
entry:
  %div = udiv <8 x i16> %0, <i16 6, i16 5, i16 6, i16 5, i16 6, i16 5, i16 6, i16 5>
  ret <8 x i16> %div
}
