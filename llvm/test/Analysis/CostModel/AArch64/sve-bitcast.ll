; RUN: opt -mtriple=aarch64-linux-gnu -mattr=+sve -passes="print<cost-model>" 2>&1 -disable-output < %s | FileCheck %s

; Integer to float bitcasts

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"

define <vscale x 2 x double> @test_nxv2f64(<vscale x 2 x i64> %a) {
; CHECK-LABEL: test_nxv2f64
; CHECK: Found an estimated cost of 0 for instruction:   %b = bitcast <vscale x 2 x i64> %a to <vscale x 2 x double>
  %b = bitcast <vscale x 2 x i64> %a to <vscale x 2 x double>
  ret <vscale x 2 x double> %b
}

define <vscale x 2 x half> @test_nxv2f16(<vscale x 2 x i16> %a) {
; CHECK-LABEL: test_nxv2f16
; CHECK: Found an estimated cost of 0 for instruction:   %b = bitcast <vscale x 2 x i16> %a to <vscale x 2 x half>
  %b = bitcast <vscale x 2 x i16> %a to <vscale x 2 x half>
  ret <vscale x 2 x half> %b
}

define <vscale x 4 x half> @test_nxv4f16(<vscale x 4 x i16> %a) {
; CHECK-LABEL: test_nxv4f16
; CHECK: Found an estimated cost of 0 for instruction:   %b = bitcast <vscale x 4 x i16> %a to <vscale x 4 x half>
  %b = bitcast <vscale x 4 x i16> %a to <vscale x 4 x half>
  ret <vscale x 4 x half> %b
}

define <vscale x 2 x float> @test_nxv2f32(<vscale x 2 x i32> %a) {
; CHECK-LABEL: test_nxv2f32
; CHECK: Found an estimated cost of 0 for instruction:   %b = bitcast <vscale x 2 x i32> %a to <vscale x 2 x float>
  %b = bitcast <vscale x 2 x i32> %a to <vscale x 2 x float>
  ret <vscale x 2 x float> %b
}

; Float to integer bitcasts

define <vscale x 2 x i64> @test_nxv2i64(<vscale x 2 x double> %a) {
; CHECK-LABEL: test_nxv2i64
; CHECK: Found an estimated cost of 0 for instruction:   %b = bitcast <vscale x 2 x double> %a to <vscale x 2 x i64>
  %b = bitcast <vscale x 2 x double> %a to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %b
}

define <vscale x 2 x i16> @test_nxv2i16(<vscale x 2 x half> %a) {
; CHECK-LABEL: test_nxv2i16
; CHECK: Found an estimated cost of 0 for instruction:   %b = bitcast <vscale x 2 x half> %a to <vscale x 2 x i16>
  %b = bitcast <vscale x 2 x half> %a to <vscale x 2 x i16>
  ret <vscale x 2 x i16> %b
}

define <vscale x 4 x i16> @test_nxv4i16(<vscale x 4 x half> %a) {
; CHECK-LABEL: test_nxv4i16
; CHECK: Found an estimated cost of 0 for instruction:   %b = bitcast <vscale x 4 x half> %a to <vscale x 4 x i16>
  %b = bitcast <vscale x 4 x half> %a to <vscale x 4 x i16>
  ret <vscale x 4 x i16> %b
}

define <vscale x 2 x i32> @test_nxv2i32(<vscale x 2 x float> %a) {
; CHECK-LABEL: test_nxv2i32
; CHECK: Found an estimated cost of 0 for instruction:   %b = bitcast <vscale x 2 x float> %a to <vscale x 2 x i32>
  %b = bitcast <vscale x 2 x float> %a to <vscale x 2 x i32>
  ret <vscale x 2 x i32> %b
}
