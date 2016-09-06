; RUN: llc -mtriple=aarch64-apple-ios %s -o - | FileCheck %s

define <16 x double> @test_sitofp_fixed(<16 x i32> %in) {
; CHECK-LABEL: test_sitofp_fixed:

  ; First, extend each i32 to i64
; CHECK-DAG: sshll2.2d [[BLOCK0_HI:v[0-9]+]], v0, #0
; CHECK-DAG: sshll2.2d [[BLOCK1_HI:v[0-9]+]], v1, #0
; CHECK-DAG: sshll2.2d [[BLOCK2_HI:v[0-9]+]], v2, #0
; CHECK-DAG: sshll2.2d [[BLOCK3_HI:v[0-9]+]], v3, #0
; CHECK-DAG: sshll.2d [[BLOCK0_LO:v[0-9]+]], v0, #0
; CHECK-DAG: sshll.2d [[BLOCK1_LO:v[0-9]+]], v1, #0
; CHECK-DAG: sshll.2d [[BLOCK2_LO:v[0-9]+]], v2, #0
; CHECK-DAG: sshll.2d [[BLOCK3_LO:v[0-9]+]], v3, #0

  ; Next, convert each to double.
; CHECK-DAG: scvtf.2d v0, [[BLOCK0_LO]]
; CHECK-DAG: scvtf.2d v1, [[BLOCK0_HI]]
; CHECK-DAG: scvtf.2d v2, [[BLOCK1_LO]]
; CHECK-DAG: scvtf.2d v3, [[BLOCK1_HI]]
; CHECK-DAG: scvtf.2d v4, [[BLOCK2_LO]]
; CHECK-DAG: scvtf.2d v5, [[BLOCK2_HI]]
; CHECK-DAG: scvtf.2d v6, [[BLOCK3_LO]]
; CHECK-DAG: scvtf.2d v7, [[BLOCK3_HI]]

; CHECK: ret
  %flt = sitofp <16 x i32> %in to <16 x double>
  %res = fdiv <16 x double> %flt, <double 64.0, double 64.0, double 64.0, double 64.0, double 64.0, double 64.0, double 64.0, double 64.0, double 64.0, double 64.0, double 64.0, double 64.0, double 64.0, double 64.0, double 64.0, double 64.0>
  ret <16 x double> %res
}

; This one is small enough to satisfy isSimple, but still illegally large.
define <4 x double> @test_sitofp_fixed_shortish(<4 x i64> %in) {
; CHECK-LABEL: test_sitofp_fixed_shortish:

; CHECK-DAG: scvtf.2d v0, v0
; CHECK-DAG: scvtf.2d v1, v1

; CHECK: ret
  %flt = sitofp <4 x i64> %in to <4 x double>
  %res = fdiv <4 x double> %flt, <double 64.0, double 64.0, double 64.0, double 64.0>
  ret <4 x double> %res
}
