; RUN: opt < %s -instcombine -S | grep "ashr i32 %val, 31"
; PR3851

define i32 @foo2(i32 %val) nounwind {
entry:
	%shr = ashr i32 %val, 15		; <i32> [#uses=3]
	%shr4 = ashr i32 %shr, 17		; <i32> [#uses=1]
        ret i32 %shr4
 }
