; RUN: llc < %s -march=x86 -mattr=-sse2,+sse | grep addps

; PR10497 + another isel issue with sse2 disabled
; (This is primarily checking that this construct doesn't crash.)
define void @a(<2 x float>* %a, <2 x i32>* %b) {
  %cc = load <2 x float>* %a
  %c = fadd <2 x float> %cc, %cc
  %dd = bitcast <2 x float> %c to <2 x i32>
  %d = add <2 x i32> %dd, %dd
  store <2 x i32> %d, <2 x i32>* %b
  ret void
}
