; RUN: llc < %s -mtriple=arm64-eabi | FileCheck %s


define <4 x i16> @fptosi_v4f64_to_v4i16(<4 x double>* %ptr) {
; CHECK-LABEL: fptosi_v4f64_to_v4i16
; CHECK-DAG: fcvtzs  v[[LHS:[0-9]+]].2d, v0.2d
; CHECK-DAG: fcvtzs  v[[RHS:[0-9]+]].2d, v1.2d
; CHECK-DAG: xtn  v[[XTN0:[0-9]+]].2s, v[[LHS]].2d
; CHECK-DAG: xtn  v[[XTN1:[0-9]+]].2s, v[[RHS]].2d
; CHECK:     uzp1  v0.4h, v[[XTN1]].4h, v[[XTN0]].4h
  %tmp1 = load <4 x double>, <4 x double>* %ptr
  %tmp2 = fptosi <4 x double> %tmp1 to <4 x i16>
  ret <4 x i16> %tmp2
}

define <8 x i8> @fptosi_v4f64_to_v4i8(<8 x double>* %ptr) {
; CHECK-LABEL: fptosi_v4f64_to_v4i8
; CHECK-DAG:  fcvtzs  v[[CONV0:[0-9]+]].2d, v0.2d
; CHECK-DAG:  fcvtzs  v[[CONV1:[0-9]+]].2d, v1.2d
; CHECK-DAG:  fcvtzs  v[[CONV2:[0-9]+]].2d, v2.2d
; CHECK-DAG:  fcvtzs  v[[CONV3:[0-9]+]].2d, v3.2d
; CHECK-DAG:  xtn  v[[XTN0:[0-9]+]].2s, v[[CONV0]].2d
; CHECK-DAG:  xtn  v[[XTN1:[0-9]+]].2s, v[[CONV1]].2d
; CHECK-DAG:  xtn  v[[XTN2:[0-9]+]].2s, v[[CONV2]].2d
; CHECK-DAG:  xtn  v[[XTN3:[0-9]+]].2s, v[[CONV3]].2d
; CHECK-DAG:  uzp1 v[[UZP0:[0-9]+]].4h, v[[XTN1]].4h, v[[XTN0]].4h
; CHECK-DAG:  uzp1 v[[UZP1:[0-9]+]].4h, v[[XTN3]].4h, v[[XTN2]].4h
; CHECK:      uzp1  v0.8b, v[[UZP1:[0-9]+]].8b, v[[UZP0:[0-9]+]].8b
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
; CHECK-LABEL: fptoui_v4f64_to_v4i16
; CHECK-DAG: fcvtzs  v[[LHS:[0-9]+]].2d, v0.2d
; CHECK-DAG: fcvtzs  v[[RHS:[0-9]+]].2d, v1.2d
; CHECK-DAG: xtn  v[[XTN0:[0-9]+]].2s, v[[LHS]].2d
; CHECK-DAG: xtn  v[[XTN1:[0-9]+]].2s, v[[RHS]].2d
; CHECK:     uzp1  v0.4h, v[[XTN1]].4h, v[[XTN0]].4h
  %tmp1 = load <4 x double>, <4 x double>* %ptr
  %tmp2 = fptoui <4 x double> %tmp1 to <4 x i16>
  ret <4 x i16> %tmp2
}
