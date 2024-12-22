; RUN: llc -march=hexagon < %s

define i1 @t_i4x8(<4 x i8> %a, <4 x i8> %b) nounwind {
entry:
	%0 = add <4 x i8> %a, %b
        %1 = bitcast <4 x i8> %0 to <32 x i1>
        %2 = extractelement <32 x i1> %1, i32 0
	ret i1 %2
}
