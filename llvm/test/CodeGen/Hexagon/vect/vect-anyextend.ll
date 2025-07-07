; RUN: llc -mtriple=hexagon < %s
; Used to fail with "Cannot select: 0x17300f0: v2i32 = any_extend"

; ModuleID = 'bugpoint-reduced-simplified.bc'
target datalayout =
"e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-a0:0-n32"
target triple = "hexagon-unknown-linux-gnu"

define void @foo() nounwind {
entry:
  %_p_vec_full48 = load <4 x i8>, ptr undef, align 8
  %0 = zext <4 x i8> %_p_vec_full48 to <4 x i32>
  store <4 x i32> %0, ptr undef, align 8
  unreachable
}
