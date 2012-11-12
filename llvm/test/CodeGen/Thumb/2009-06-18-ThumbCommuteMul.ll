; RUN: llc < %s -march=thumb | grep r0 | count 1

define i32 @a(i32 %x, i32 %y) nounwind readnone {
entry:
	%mul = mul i32 %y, %x		; <i32> [#uses=1]
	ret i32 %mul
}

