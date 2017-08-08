; Test multiple return values (LLVM ABI extension)
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Up to eight vector return values fit into VRs.
define { <2 x double>, <2 x double>, <2 x double>, <2 x double>,
         <2 x double>, <2 x double>, <2 x double>, <2 x double> } @f1() {
; CHECK-LABEL: f1:
; CHECK: larl [[TMP:%r[0-5]]], .LCPI
; CHECK: vl %v24, 0([[TMP]])
; CHECK: larl [[TMP:%r[0-5]]], .LCPI
; CHECK: vl %v26, 0([[TMP]])
; CHECK: larl [[TMP:%r[0-5]]], .LCPI
; CHECK: vl %v28, 0([[TMP]])
; CHECK: larl [[TMP:%r[0-5]]], .LCPI
; CHECK: vl %v30, 0([[TMP]])
; CHECK: larl [[TMP:%r[0-5]]], .LCPI
; CHECK: vl %v25, 0([[TMP]])
; CHECK: larl [[TMP:%r[0-5]]], .LCPI
; CHECK: vl %v27, 0([[TMP]])
; CHECK: larl [[TMP:%r[0-5]]], .LCPI
; CHECK: vl %v29, 0([[TMP]])
; CHECK: larl [[TMP:%r[0-5]]], .LCPI
; CHECK: vl %v31, 0([[TMP]])
; CHECK: br %r14
  ret { <2 x double>, <2 x double>, <2 x double>, <2 x double>,
        <2 x double>, <2 x double>, <2 x double>, <2 x double> }
      { <2 x double> <double 1.0, double 1.1>,
        <2 x double> <double 2.0, double 2.1>,
        <2 x double> <double 3.0, double 3.1>,
        <2 x double> <double 4.0, double 4.1>,
        <2 x double> <double 5.0, double 5.1>,
        <2 x double> <double 6.0, double 6.1>,
        <2 x double> <double 7.0, double 7.1>,
        <2 x double> <double 8.0, double 8.1> }
}

; More than eight vector return values use sret.
define { <2 x double>, <2 x double>, <2 x double>, <2 x double>,
         <2 x double>, <2 x double>, <2 x double>, <2 x double>,
         <2 x double> } @f2() {
; CHECK-LABEL: f2:
; CHECK: larl [[TMP:%r[0-5]]], .LCPI
; CHECK: vl [[VTMP:%v[0-9]+]], 0([[TMP]])
; CHECK-DAG: vst [[VTMP]], 128(%r2)
; CHECK-DAG: larl [[TMP:%r[0-5]]], .LCPI
; CHECK: vl [[VTMP:%v[0-9]+]], 0([[TMP]])
; CHECK-DAG: vst [[VTMP]], 112(%r2)
; CHECK-DAG: larl [[TMP:%r[0-5]]], .LCPI
; CHECK: vl [[VTMP:%v[0-9]+]], 0([[TMP]])
; CHECK-DAG: vst [[VTMP]], 96(%r2)
; CHECK-DAG: larl [[TMP:%r[0-5]]], .LCPI
; CHECK: vl [[VTMP:%v[0-9]+]], 0([[TMP]])
; CHECK-DAG: vst [[VTMP]], 80(%r2)
; CHECK-DAG: larl [[TMP:%r[0-5]]], .LCPI
; CHECK: vl [[VTMP:%v[0-9]+]], 0([[TMP]])
; CHECK-DAG: vst [[VTMP]], 64(%r2)
; CHECK-DAG: larl [[TMP:%r[0-5]]], .LCPI
; CHECK: vl [[VTMP:%v[0-9]+]], 0([[TMP]])
; CHECK-DAG: vst [[VTMP]], 48(%r2)
; CHECK-DAG: larl [[TMP:%r[0-5]]], .LCPI
; CHECK: vl [[VTMP:%v[0-9]+]], 0([[TMP]])
; CHECK-DAG: vst [[VTMP]], 32(%r2)
; CHECK-DAG: larl [[TMP:%r[0-5]]], .LCPI
; CHECK: vl [[VTMP:%v[0-9]+]], 0([[TMP]])
; CHECK-DAG: vst [[VTMP]], 16(%r2)
; CHECK-DAG: larl [[TMP:%r[0-5]]], .LCPI
; CHECK: vl [[VTMP:%v[0-9]+]], 0([[TMP]])
; CHECK: vst [[VTMP]], 0(%r2)
; CHECK: br %r14
  ret { <2 x double>, <2 x double>, <2 x double>, <2 x double>,
        <2 x double>, <2 x double>, <2 x double>, <2 x double>,
        <2 x double> }
      { <2 x double> <double 1.0, double 1.1>,
        <2 x double> <double 2.0, double 2.1>,
        <2 x double> <double 3.0, double 3.1>,
        <2 x double> <double 4.0, double 4.1>,
        <2 x double> <double 5.0, double 5.1>,
        <2 x double> <double 6.0, double 6.1>,
        <2 x double> <double 7.0, double 7.1>,
        <2 x double> <double 8.0, double 8.1>,
        <2 x double> <double 9.0, double 9.1> }
}
