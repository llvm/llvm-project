; RUN: llc -march=hexagon -mattr=+hvxv73,+hvx-length128b %s -o - | FileCheck %s
; RUN: llc -march=hexagon -mattr=+hvxv73,+hvx-length128b -debug-only=isel \
; RUN:   -filter-print-funcs=reduce_2d_vec,reduce_2d_pair,reduce_maximum_vec,reduce_maximum_pair \
; RUN:   %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ISEL
; REQUIRES: asserts

define dso_local float @reduce_2d_vec(ptr noundef readonly captures(none) %X) local_unnamed_addr {
entry:
  %0 = load <32 x float>, ptr %X, align 4
  %.ripple.reduction = tail call reassoc float @llvm.vector.reduce.fmax.v32f32(<32 x float> %0)
  ret float %.ripple.reduction
}
;CHECK: reduce_2d_vec
;CHECK: vmax
;CHECK: vmax
;CHECK: vmax
;CHECK: vmax
;CHECK: vmax
;CHECK: .Lfunc_end0
; ISEL-LABEL: === reduce_2d_vec
; ISEL: v32f32 = fmaximum

define dso_local float @reduce_2d_pair(ptr noundef readonly captures(none) %X) local_unnamed_addr {
entry:
  %0 = load <64 x float>, ptr %X, align 4
  %.ripple.reduction = tail call reassoc float @llvm.vector.reduce.fmax.v64f32(<64 x float> %0)
  ret float %.ripple.reduction
}
;CHECK: reduce_2d_pair
;CHECK: vmax
;CHECK: vmax
;CHECK: vmax
;CHECK: vmax
;CHECK: vmax
;CHECK: vmax
;CHECK: .Lfunc_end1
; ISEL-LABEL: === reduce_2d_pair
; ISEL: v32f32 = fmaximum

define dso_local float @reduce_maximum_vec(ptr noundef readonly captures(none) %X) local_unnamed_addr {
entry:
  %0 = load <32 x float>, ptr %X, align 4
  %.ripple.reduction = tail call reassoc float @llvm.vector.reduce.fmaximum.v32f32(<32 x float> %0)
  ret float %.ripple.reduction
}
;CHECK: reduce_maximum_vec
;CHECK: vmax
;CHECK: vmax
;CHECK: vmax
;CHECK: vmax
;CHECK: vmax
;CHECK: .Lfunc_end2
; ISEL-LABEL: === reduce_maximum_vec
; ISEL: v32f32 = fmaximum

define dso_local float @reduce_maximum_pair(ptr noundef readonly captures(none) %X) local_unnamed_addr {
entry:
  %0 = load <64 x float>, ptr %X, align 4
  %.ripple.reduction = tail call reassoc float @llvm.vector.reduce.fmaximum.v64f32(<64 x float> %0)
  ret float %.ripple.reduction
}
;CHECK: reduce_maximum_pair
;CHECK: vmax
;CHECK: vmax
;CHECK: vmax
;CHECK: vmax
;CHECK: vmax
;CHECK: vmax
;CHECK: .Lfunc_end3
; ISEL-LABEL: === reduce_maximum_pair
; ISEL: v32f32 = fmaximum
