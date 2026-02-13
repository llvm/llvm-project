; RUN: llc -mtriple=hexagon -mcpu=hexagonv73 -mattr=+hvxv73,+hvx-length128b \
; RUN:   -enable-legalize-types-checking < %s | FileCheck %s
;
; Check that HVX zero-extend operations that require splitting compile
; correctly. The bug was that multi-step TL_EXTEND operations (e.g., i8->i32)
; were split directly, creating sub-HVX operand types (v64i8) that confused
; the legalizer's map tracking. The fix expands multi-step extends into
; single steps first.

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon-unknown-linux-musl"

; CHECK-LABEL: test_zext_split:
; CHECK-DAG:   vunpack
; CHECK-DAG:   vadd
; CHECK:       jumpr r31
define fastcc <32 x i32> @test_zext_split(<32 x i8> %a, <32 x i8> %b) {
entry:
  %ext_a = zext <32 x i8> %a to <32 x i32>
  %ext_b = zext <32 x i8> %b to <32 x i32>
  %sum = add <32 x i32> %ext_a, %ext_b
  ret <32 x i32> %sum
}
