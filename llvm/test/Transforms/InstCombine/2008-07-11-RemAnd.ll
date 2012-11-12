; RUN: opt < %s -instcombine -S | not grep rem
; PR2330

define i32 @a(i32 %b) nounwind  {
entry:
	srem i32 %b, 8		; <i32>:0 [#uses=1]
	and i32 %0, 1		; <i32>:1 [#uses=1]
	ret i32 %1
}
