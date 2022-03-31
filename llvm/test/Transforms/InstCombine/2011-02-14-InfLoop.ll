; This testcase causes an infinite loop in the instruction combiner,
; because it changes a pattern and the original pattern is almost
; identical to the newly-generated pattern.
; RUN: opt < %s -passes=instcombine -disable-output

;PR PR9216

target triple = "x86_64-unknown-linux-gnu"

define <4 x float> @m_387(i8* noalias nocapture %A, i8* nocapture %B, <4 x i1> %C) nounwind {
entry:
  %movcsext20 = sext <4 x i1> %C to <4 x i32>
  %tmp2389 = xor <4 x i32> %movcsext20, <i32 -1, i32 -1, i32 -1, i32 -1>
  %movcand25 = and <4 x i32> %tmp2389, <i32 undef, i32 undef, i32 undef, i32 -1>
  %movcor26 = or <4 x i32> %movcand25, zeroinitializer
  %L2 = bitcast <4 x i32> %movcor26 to <4 x float>
  %L3 = shufflevector <4 x float> zeroinitializer, <4 x float> %L2, <4 x i32> <i32 0, i32 1, i32 2, i32 7>
  ret <4 x float> %L3
}
