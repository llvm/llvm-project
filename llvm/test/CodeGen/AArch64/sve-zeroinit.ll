; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; WARN-NOT: warning

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

define <vscale x 2 x i1> @test_zeroinit_2xi1() {
; CHECK-LABEL: test_zeroinit_2xi1
; CHECK:       whilelo p0.d, xzr, xzr
; CHECK-NEXT:  ret
  ret <vscale x 2 x i1> zeroinitializer
}

define <vscale x 4 x i1> @test_zeroinit_4xi1() {
; CHECK-LABEL: test_zeroinit_4xi1
; CHECK:       whilelo p0.s, xzr, xzr
; CHECK-NEXT:  ret
  ret <vscale x 4 x i1> zeroinitializer
}

define <vscale x 8 x i1> @test_zeroinit_8xi1() {
; CHECK-LABEL: test_zeroinit_8xi1
; CHECK:       whilelo p0.h, xzr, xzr
; CHECK-NEXT:  ret
  ret <vscale x 8 x i1> zeroinitializer
}

define <vscale x 16 x i1> @test_zeroinit_16xi1() {
; CHECK-LABEL: test_zeroinit_16xi1
; CHECK:       whilelo p0.b, xzr, xzr
; CHECK-NEXT:  ret
  ret <vscale x 16 x i1> zeroinitializer
}
