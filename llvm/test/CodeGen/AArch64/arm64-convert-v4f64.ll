; RUN: llc < %s -mtriple=aarch64-none-linux-gnu -march=arm64 | FileCheck %s


define <4 x i16> @fptosi_v4f64_to_v4i16(<4 x double>* %ptr) {
; CHECK: fptosi_v4f64_to_v4i16
; CHECK-DAG: fcvtzs  v[[LHS:[0-9]+]].2d, v0.2d
; CHECK-DAG: fcvtzs  v[[RHS:[0-9]+]].2d, v1.2d
; CHECK-DAG: xtn  v[[MID:[0-9]+]].2s, v[[LHS]].2d
; CHECK-DAG: xtn2  v[[MID]].4s, v[[RHS]].2d
; CHECK:     xtn  v0.4h, v[[MID]].4s
  %tmp1 = load <4 x double>, <4 x double>* %ptr
  %tmp2 = fptosi <4 x double> %tmp1 to <4 x i16>
  ret <4 x i16> %tmp2
}

define <8 x i8> @fptosi_v4f64_to_v4i8(<8 x double>* %ptr) {
; CHECK: fptosi_v4f64_to_v4i8
; CHECK-DAG:  fcvtzs  v[[CONV0:[0-9]+]].2d, v0.2d
; CHECK-DAG:  fcvtzs  v[[CONV1:[0-9]+]].2d, v1.2d
; CHECK-DAG:  fcvtzs  v[[CONV2:[0-9]+]].2d, v2.2d
; CHECK-DAG:  fcvtzs  v[[CONV3:[0-9]+]].2d, v3.2d
; CHECK-DAG:  xtn  v[[NA2:[0-9]+]].2s, v[[CONV2]].2d
; CHECK-DAG:  xtn2  v[[NA2]].4s, v[[CONV3]].2d
; CHECK-DAG:  xtn  v[[NA0:[0-9]+]].2s, v[[CONV0]].2d
; CHECK-DAG:  xtn2  v[[NA0]].4s, v[[CONV1]].2d
; CHECK-DAG:  xtn  v[[TMP1:[0-9]+]].4h, v[[NA2]].4s
; CHECK-DAG:  xtn2  v[[TMP1]].8h, v[[NA0]].4s
; CHECK:      xtn  v0.8b, v[[TMP1]].8h
  %tmp1 = load <8 x double>, <8 x double>* %ptr
  %tmp2 = fptosi <8 x double> %tmp1 to <8 x i8>
  ret <8 x i8> %tmp2
}

define <4 x half> @uitofp_v4i64_to_v4f16(<4 x i64>* %ptr) {
; CHECK: uitofp_v4i64_to_v4f16
; CHECK-DAG: ucvtf  v[[LHS:[0-9]+]].2d, v0.2d
; CHECK-DAG: ucvtf  v[[RHS:[0-9]+]].2d, v1.2d
; CHECK-DAG: fcvtn  v[[MID:[0-9]+]].2s, v[[LHS]].2d
; CHECK-DAG: fcvtn2  v[[MID]].4s, v[[RHS]].2d
; CHECK:     fcvtn  v0.4h, v[[MID]].4s
  %tmp1 = load <4 x i64>, <4 x i64>* %ptr
  %tmp2 = uitofp <4 x i64> %tmp1 to <4 x half>
  ret <4 x half> %tmp2
}

define <4 x i16> @trunc_v4i64_to_v4i16(<4 x i64>* %ptr) {
; CHECK: trunc_v4i64_to_v4i16
; CHECK: xtn
; CHECK: xtn2
; CHECK: xtn
  %tmp1 = load <4 x i64>, <4 x i64>* %ptr
  %tmp2 = trunc <4 x i64> %tmp1 to <4 x i16>
  ret <4 x i16> %tmp2
}

define <4 x i16> @fptoui_v4f64_to_v4i16(<4 x double>* %ptr) {
; CHECK: fptoui_v4f64_to_v4i16
; CHECK-DAG: fcvtzu  v[[LHS:[0-9]+]].2d, v0.2d
; CHECK-DAG: fcvtzu  v[[RHS:[0-9]+]].2d, v1.2d
; CHECK-DAG: xtn  v[[MID:[0-9]+]].2s, v[[LHS]].2d
; CHECK-DAG: xtn2  v[[MID]].4s, v[[RHS]].2d
; CHECK:     xtn  v0.4h, v[[MID]].4s
  %tmp1 = load <4 x double>, <4 x double>* %ptr
  %tmp2 = fptoui <4 x double> %tmp1 to <4 x i16>
  ret <4 x i16> %tmp2
}
