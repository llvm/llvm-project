; RUN: opt < %s -passes=instcombine -S | grep -v "i32 8"
; PR2330

define i32 @a(i32 %a) nounwind  {
entry:
	%tmp2 = sub i32 8, %a		; <i32> [#uses=1]
	%tmp3 = and i32 %tmp2, 7		; <i32> [#uses=1]
	ret i32 %tmp3
}
