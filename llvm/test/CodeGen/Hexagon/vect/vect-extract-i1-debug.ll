; RUN: llc -mtriple=hexagon -debug-only=isel < %s 2>/dev/null
; REQUIRES: asserts

; Make sure that this doesn't crash. Debug option enabled a failing assertion
; about type mismatch in formal arguments.
; CHECK: vaddub

define i1 @t_i4x8(<4 x i8> %a, <4 x i8> %b) nounwind {
entry:
	%0 = add <4 x i8> %a, %b
        %1 = bitcast <4 x i8> %0 to <32 x i1>
        %2 = extractelement <32 x i1> %1, i32 0
	ret i1 %2
}
