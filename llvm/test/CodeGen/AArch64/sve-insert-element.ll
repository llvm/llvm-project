; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s

define <vscale x 16 x i8> @test_lane0_16xi8(<vscale x 16 x i8> %a) {
; CHECK-LABEL: test_lane0_16xi8
; CHECK:       mov [[REG:.*]], #30
; CHECK:       mov z0.b, p{{[0-7]}}/m, [[REG]]
  %b = insertelement <vscale x 16 x i8> %a, i8 30, i32 0
  ret <vscale x 16 x i8> %b
}

define <vscale x 8 x i16> @test_lane0_8xi16(<vscale x 8 x i16> %a) {
; CHECK-LABEL: test_lane0_8xi16
; CHECK:       mov [[REG:.*]], #30
; CHECK:       mov z0.h, p{{[0-7]}}/m, [[REG]]
  %b = insertelement <vscale x 8 x i16> %a, i16 30, i32 0
  ret <vscale x 8 x i16> %b
}

define <vscale x 4 x i32> @test_lane0_4xi32(<vscale x 4 x i32> %a) {
; CHECK-LABEL: test_lane0_4xi32
; CHECK:       mov [[REG:.*]], #30
; CHECK:       mov z0.s, p{{[0-7]}}/m, [[REG]]
  %b = insertelement <vscale x 4 x i32> %a, i32 30, i32 0
  ret <vscale x 4 x i32> %b
}

define <vscale x 2 x i64> @test_lane0_2xi64(<vscale x 2 x i64> %a) {
; CHECK-LABEL: test_lane0_2xi64
; CHECK:       mov w[[REG:.*]], #30
; CHECK:       mov z0.d, p{{[0-7]}}/m, x[[REG]]
  %b = insertelement <vscale x 2 x i64> %a, i64 30, i32 0
  ret <vscale x 2 x i64> %b
}

define <vscale x 2 x double> @test_lane0_2xf64(<vscale x 2 x double> %a) {
; CHECK-LABEL: test_lane0_2xf64
; CHECK:       fmov d[[REG:[0-9]+]], #1.00000000
; CHECK:       mov z0.d, p{{[0-7]}}/m, z[[REG]].d
  %b = insertelement <vscale x 2 x double> %a, double 1.0, i32 0
  ret <vscale x 2 x double> %b
}

define <vscale x 4 x float> @test_lane0_4xf32(<vscale x 4 x float> %a) {
; CHECK-LABEL: test_lane0_4xf32
; CHECK:       fmov s[[REG:[0-9]+]], #1.00000000
; CHECK:       mov z0.s, p{{[0-7]}}/m, z[[REG]].s
  %b = insertelement <vscale x 4 x float> %a, float 1.0, i32 0
  ret <vscale x 4 x float> %b
}

define <vscale x 8 x half> @test_lane0_8xf16(<vscale x 8 x half> %a) {
; CHECK-LABEL: test_lane0_8xf16
; CHECK:       fmov h[[REG:[0-9]+]], #1.00000000
; CHECK:       mov z0.h, p{{[0-7]}}/m, z[[REG]].h
  %b = insertelement <vscale x 8 x half> %a, half 1.0, i32 0
  ret <vscale x 8 x half> %b
}

; Undefined lane insert
define <vscale x 2 x i64> @test_lane4_2xi64(<vscale x 2 x i64> %a) {
; CHECK-LABEL: test_lane4_2xi64
; CHECK:       mov w[[IDXREG:.*]], #4
; CHECK:       index z[[CMPVEC:[0-9]+]].d, #0, #1
; CHECK:       mov z[[IDXVEC:[0-9]+]].d, x[[IDXREG]]
; CHECK:       cmpeq p[[PRED:[0-9]+]].d, p{{[0-7]}}/z, z[[CMPVEC]].d, z[[IDXVEC]].d
; CHECK:       mov w[[VALREG:.*]], #30
; CHECK:       mov z0.d, p[[PRED]]/m, x[[VALREG]]
  %b = insertelement <vscale x 2 x i64> %a, i64 30, i32 4
  ret <vscale x 2 x i64> %b
}

; Undefined lane insert
define <vscale x 8 x half> @test_lane9_8xf16(<vscale x 8 x half> %a) {
; CHECK-LABEL: test_lane9_8xf16
; CHECK:       mov w[[IDXREG:.*]], #9
; CHECK:       index z[[CMPVEC:[0-9]+]].h, #0, #1
; CHECK:       mov z[[IDXVEC:[0-9]+]].h, w[[IDXREG]]
; CHECK:       cmpeq p[[PRED:[0-9]+]].h, p{{[0-7]}}/z, z[[CMPVEC]].h, z[[IDXVEC]].h
; CHECK:       fmov h[[VALREG:[0-9]+]], #1.00000000
; CHECK:       mov z0.h, p[[PRED]]/m, h[[VALREG]]
  %b = insertelement <vscale x 8 x half> %a, half 1.0, i32 9
  ret <vscale x 8 x half> %b
}

define <vscale x 16 x i8> @test_lane1_16xi8(<vscale x 16 x i8> %a) {
; CHECK-LABEL: test_lane1_16xi8
; CHECK:       mov w[[IDXREG:.*]], #1
; CHECK:       index z[[CMPVEC:[0-9]+]].b, #0, #1
; CHECK:       mov z[[IDXVEC:[0-9]+]].b, w[[IDXREG]]
; CHECK:       cmpeq p[[PRED:[0-9]+]].b, p{{[0-7]}}/z, z[[CMPVEC]].b, z[[IDXVEC]].b
; CHECK:       mov w[[VALREG:.*]], #30
; CHECK:       mov z0.b, p[[PRED]]/m, w[[VALREG]]
  %b = insertelement <vscale x 16 x i8> %a, i8 30, i32 1
  ret <vscale x 16 x i8> %b
}

define <vscale x 16 x i8> @test_lanex_16xi8(<vscale x 16 x i8> %a, i32 %x) {
; CHECK-LABEL: test_lanex_16xi8
; CHECK:       index z[[CMPVEC:[0-9]+]].b, #0, #1
; CHECK:       mov z[[IDXVEC:[0-9]+]].b, w[[IDXREG]]
; CHECK:       cmpeq p[[PRED:[0-9]+]].b, p{{[0-7]}}/z, z[[CMPVEC]].b, z[[IDXVEC]].b
; CHECK:       mov w[[VALREG:.*]], #30
; CHECK:       mov z0.b, p[[PRED]]/m, w[[VALREG]]
  %b = insertelement <vscale x 16 x i8> %a, i8 30, i32 %x
  ret <vscale x 16 x i8> %b
}


; Redundant lane insert
define <vscale x 4 x i32> @extract_insert_4xi32(<vscale x 4 x i32> %a) {
; CHECK-LABEL: extract_insert_4xi32
; CHECK-NOT:   mov w{{.*}}, #30
; CHECK-NOT:   mov z0.d
  %b = extractelement <vscale x 4 x i32> %a, i32 2
  %c = insertelement <vscale x 4 x i32> %a, i32 %b, i32 2
  ret <vscale x 4 x i32> %c
}

define <vscale x 8 x i16> @test_lane6_undef_8xi16(i16 %a) {
; CHECK-LABEL: test_lane6_undef_8xi16
; CHECK:       mov w[[IDXREG:.*]], #6
; CHECK:       index z[[CMPVEC:.*]].h, #0, #1
; CHECK:       mov z[[IDXVEC:[0-9]+]].h, w[[IDXREG]]
; CHECK:       cmpeq p[[PRED:.*]].h, p{{.*}}/z, z[[CMPVEC]].h, z[[IDXVEC]].h
; CHECK:       mov z0.h, p[[PRED]]/m, w0
  %b = insertelement <vscale x 8 x i16> undef, i16 %a, i32 6
  ret <vscale x 8 x i16> %b
}

define <vscale x 16 x i8> @test_lane0_undef_16xi8(i8 %a) {
; CHECK-LABEL: test_lane0_undef_16xi8
; CHECK:       fmov s0, w0
  %b = insertelement <vscale x 16 x i8> undef, i8 %a, i32 0
  ret <vscale x 16 x i8> %b
}
