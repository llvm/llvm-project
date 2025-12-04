; RUN: llc -mtriple=hexagon -mattr=+hvx,+hvx-length128b,+hvxv81 %s -o - | FileCheck %s

; Test that on hexagon computation of a_sq_f32 is done using a widening multiply
; instruction.
define dso_local <64 x float> @a_sq_times_b_sq(<64 x half>  %a, <64 x float> %b) {
entry:
  %a_sq_f16 = fmul <64 x half> %a, %a
  %a_sq_f32 = fpext <64 x half> %a_sq_f16 to <64 x float>
  %b_sq = fmul <64 x float> %b, %b
  %result = fmul <64 x float> %a_sq_f32, %b_sq
  ret <64 x float> %result
}
; CHECK: a_sq_times_b_sq
; CHECK: .qf32 = vmpy({{v[0-9]+}}.hf,{{v[0-9]+}}.hf)
; CHECK: .qf32 = vmpy({{v[0-9]+}}.sf,{{v[0-9]+}}.sf)
; CHECK: .qf32 = vmpy({{v[0-9]+}}.sf,{{v[0-9]+}}.sf)
