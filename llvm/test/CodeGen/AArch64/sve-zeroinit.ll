; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-none-linux-gnu"

define <vscale x 2 x i64> @test_zeroinit_2xi64() {
; CHECK-LABEL: test_zeroinit_2xi64
; CHECK:       mov z0.d, #0
; CHECK-NEXT:  ret
  ret <vscale x 2 x i64> zeroinitializer
}

define <vscale x 4 x i32> @test_zeroinit_4xi32() {
; CHECK-LABEL: test_zeroinit_4xi32
; CHECK:       mov z0.s, #0
; CHECK-NEXT:  ret
  ret <vscale x 4 x i32> zeroinitializer
}

define <vscale x 8 x i16> @test_zeroinit_8xi16() {
; CHECK-LABEL: test_zeroinit_8xi16
; CHECK:       mov z0.h, #0
; CHECK-NEXT:  ret
  ret <vscale x 8 x i16> zeroinitializer
}

define <vscale x 16 x i8> @test_zeroinit_16xi8() {
; CHECK-LABEL: test_zeroinit_16xi8
; CHECK:       mov z0.b, #0
; CHECK-NEXT:  ret
  ret <vscale x 16 x i8> zeroinitializer
}

define <vscale x 2 x double> @test_zeroinit_2xf64() {
; CHECK-LABEL: test_zeroinit_2xf64
; CHECK:       mov z0.d, #0
; CHECK-NEXT:  ret
  ret <vscale x 2 x double> zeroinitializer
}

define <vscale x 4 x float> @test_zeroinit_4xf32() {
; CHECK-LABEL: test_zeroinit_4xf32
; CHECK:       mov z0.s, #0
; CHECK-NEXT:  ret
  ret <vscale x 4 x float> zeroinitializer
}

define <vscale x 8 x half> @test_zeroinit_8xf16() {
; CHECK-LABEL: test_zeroinit_8xf16
; CHECK:       mov z0.h, #0
; CHECK-NEXT:  ret
  ret <vscale x 8 x half> zeroinitializer
}

define <vscale x 1 x i1> @test_zeroinit_1xi1() {
; CHECK-LABEL: test_zeroinit_1xi1
; CHECK:       pfalse p0.b
; CHECK-NEXT:  ret
  ret <vscale x 1 x i1> zeroinitializer
}

define <vscale x 2 x i1> @test_zeroinit_2xi1() {
; CHECK-LABEL: test_zeroinit_2xi1
; CHECK:       pfalse p0.b
; CHECK-NEXT:  ret
  ret <vscale x 2 x i1> zeroinitializer
}

define <vscale x 4 x i1> @test_zeroinit_4xi1() {
; CHECK-LABEL: test_zeroinit_4xi1
; CHECK:       pfalse p0.b
; CHECK-NEXT:  ret
  ret <vscale x 4 x i1> zeroinitializer
}

define <vscale x 8 x i1> @test_zeroinit_8xi1() {
; CHECK-LABEL: test_zeroinit_8xi1
; CHECK:       pfalse p0.b
; CHECK-NEXT:  ret
  ret <vscale x 8 x i1> zeroinitializer
}

define <vscale x 16 x i1> @test_zeroinit_16xi1() {
; CHECK-LABEL: test_zeroinit_16xi1
; CHECK:       pfalse p0.b
; CHECK-NEXT:  ret
  ret <vscale x 16 x i1> zeroinitializer
}

define target("aarch64.svcount") @test_zeroinit_svcount() "target-features"="+sme2" {
; CHECK-LABEL: test_zeroinit_svcount
; CHECK:       pfalse p0.b
; CHECK-NEXT:  ret
  ret target("aarch64.svcount") zeroinitializer
}
