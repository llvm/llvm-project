; RUN: opt < %s -passes=instcombine -S | not grep mul

define i26 @test(i26 %a) nounwind  {
entry:
	%_add = mul i26 %a, 2885		; <i26> [#uses=1]
	%_shl2 = mul i26 %a, 2884		; <i26> [#uses=1]
	%_sub = sub i26 %_add, %_shl2		; <i26> [#uses=1]
	ret i26 %_sub
}
