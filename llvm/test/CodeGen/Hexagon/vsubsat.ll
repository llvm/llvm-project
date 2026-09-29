; RUN: llc -march=hexagon -mattr=+hvxv73,+hvx-length128b < %s | FileCheck %s

;; Saturating subtraction.

; CHECK-LABEL: vsububsat
; CHECK: v[[#]].ub = vsub(v[[#]].ub,v[[#]].ub):sat
define dso_local <128 x i8> @vsububsat(<128 x i8> %x, <128 x i8> %y) {
entry:
  %0 = tail call <128 x i8> @llvm.usub.sat.v128i8(<128 x i8> %x, <128 x i8> %y)
  ret <128 x i8> %0
}

; CHECK-LABEL: vsubuhsat
; CHECK: v[[#]].uh = vsub(v[[#]].uh,v[[#]].uh):sat
define dso_local <64 x i16> @vsubuhsat(<64 x i16> %x, <64 x i16> %y) {
entry:
  %0 = tail call <64 x i16> @llvm.usub.sat.v64i16(<64 x i16> %x, <64 x i16> %y)
  ret <64 x i16> %0
}

; CHECK-LABEL: vsubuwsat
; CHECK: v[[#]].uw = vsub(v[[#]].uw,v[[#]].uw):sat
define dso_local <32 x i32> @vsubuwsat(<32 x i32> %x, <32 x i32> %y) {
entry:
  %0 = tail call <32 x i32> @llvm.usub.sat.v32i32(<32 x i32> %x, <32 x i32> %y)
  ret <32 x i32> %0
}

; CHECK-LABEL: vsubbsat
; CHECK: v[[#]].b = vsub(v[[#]].b,v[[#]].b):sat
define dso_local <128 x i8> @vsubbsat(<128 x i8> %x, <128 x i8> %y) {
entry:
  %0 = tail call <128 x i8> @llvm.ssub.sat.v128i8(<128 x i8> %x, <128 x i8> %y)
  ret <128 x i8> %0
}

; CHECK-LABEL: vsubhsat
; CHECK: v[[#]].h = vsub(v[[#]].h,v[[#]].h):sat
define dso_local <64 x i16> @vsubhsat(<64 x i16> %x, <64 x i16> %y) {
entry:
  %0 = tail call <64 x i16> @llvm.ssub.sat.v64i16(<64 x i16> %x, <64 x i16> %y)
  ret <64 x i16> %0
}

; CHECK-LABEL: vsubwsat
; CHECK: v[[#]].w = vsub(v[[#]].w,v[[#]].w):sat
define dso_local <32 x i32> @vsubwsat(<32 x i32> %x, <32 x i32> %y) {
entry:
  %0 = tail call <32 x i32> @llvm.ssub.sat.v32i32(<32 x i32> %x, <32 x i32> %y)
  ret <32 x i32> %0
}

; CHECK-LABEL: vsububsat_dv
; CHECK: v[[#]]:[[#]].ub = vsub(v[[#]]:[[#]].ub,v[[#]]:[[#]].ub):sat
define dso_local <256 x i8> @vsububsat_dv(<256 x i8> %x, <256 x i8> %y) {
entry:
  %0 = tail call <256 x i8> @llvm.usub.sat.v256i8(<256 x i8> %x, <256 x i8> %y)
  ret <256 x i8> %0
}

; CHECK-LABEL: vsubuhsat_dv
; CHECK: v[[#]]:[[#]].uh = vsub(v[[#]]:[[#]].uh,v[[#]]:[[#]].uh):sat
define dso_local <128 x i16> @vsubuhsat_dv(<128 x i16> %x, <128 x i16> %y) {
entry:
  %0 = tail call <128 x i16> @llvm.usub.sat.v128i16(<128 x i16> %x, <128 x i16> %y)
  ret <128 x i16> %0
}

; CHECK-LABEL: vsubuwsat_dv
; CHECK: v[[#]]:[[#]].uw = vsub(v[[#]]:[[#]].uw,v[[#]]:[[#]].uw):sat
define dso_local <64 x i32> @vsubuwsat_dv(<64 x i32> %x, <64 x i32> %y) {
entry:
  %0 = tail call <64 x i32> @llvm.usub.sat.v64i32(<64 x i32> %x, <64 x i32> %y)
  ret <64 x i32> %0
}

; CHECK-LABEL: vsubbsat_dv
; CHECK: v[[#]]:[[#]].b = vsub(v[[#]]:[[#]].b,v[[#]]:[[#]].b):sat
define dso_local <256 x i8> @vsubbsat_dv(<256 x i8> %x, <256 x i8> %y) {
entry:
  %0 = tail call <256 x i8> @llvm.ssub.sat.v256i8(<256 x i8> %x, <256 x i8> %y)
  ret <256 x i8> %0
}

; CHECK-LABEL: vsubhsat_dv
; CHECK: v[[#]]:[[#]].h = vsub(v[[#]]:[[#]].h,v[[#]]:[[#]].h):sat
define dso_local <128 x i16> @vsubhsat_dv(<128 x i16> %x, <128 x i16> %y) {
entry:
  %0 = tail call <128 x i16> @llvm.ssub.sat.v128i16(<128 x i16> %x, <128 x i16> %y)
  ret <128 x i16> %0
}

; CHECK-LABEL: vsubwsat_dv
; CHECK: v[[#]]:[[#]].w = vsub(v[[#]]:[[#]].w,v[[#]]:[[#]].w):sat
define dso_local <64 x i32> @vsubwsat_dv(<64 x i32> %x, <64 x i32> %y) {
entry:
  %0 = tail call <64 x i32> @llvm.ssub.sat.v64i32(<64 x i32> %x, <64 x i32> %y)
  ret <64 x i32> %0
}
