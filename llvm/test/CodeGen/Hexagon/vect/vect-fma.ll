; RUN: llc -march=hexagon -mcpu=hexagonv5 < %s
; REQUIRES: asserts
; Used to fail with "SplitVectorResult #0: 0x16cbe60: v4f64 = fma"

; ModuleID = 'bugpoint-reduced-simplified.bc'
target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-a0:0-n32"
target triple = "hexagon-unknown-linux-gnu"

define void @run() nounwind {
entry:
  br label %polly.loop_header

polly.loop_after:                                 ; preds = %polly.loop_header
  ret void

polly.loop_header:                                ; preds = %polly.loop_body, %entry
  %0 = icmp sle i32 undef, 399
  br i1 %0, label %polly.loop_body, label %polly.loop_after

polly.loop_body:                                  ; preds = %polly.loop_header
  %_p_vec_full = load <4 x double>, ptr undef, align 8
  %mulp_vec = fmul <4 x double> %_p_vec_full, <double 7.000000e+00, double 7.000000e+00, double 7.000000e+00, double 7.000000e+00>
  %addp_vec = fadd <4 x double> undef, %mulp_vec
  store <4 x double> %addp_vec, ptr undef, align 8
  br label %polly.loop_header
}
