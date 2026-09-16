; RUN: llc -mtriple=hexagon -mcpu=hexagonv81 -mattr=+hvxv81,+hvx-length128b,+hvx-ieee-fp,-packets < %s | FileCheck %s

; Before the fix, V81 processor definition in Hexagon.td was missing early
; architecture features (ArchV5, ArchV55, ArchV60, ArchV62), causing compilation
; to fail with:
;   "fatal error: Attempting to emit S6_vsplatrbp instruction but the
;    Feature_HasV62 predicate(s) are not met"

; CHECK-LABEL: test_hvx_v16i8_return:
; CHECK: vsplatb

define <16 x i8> @test_hvx_v16i8_return() {
entry:
  ret <16 x i8> zeroinitializer
}

