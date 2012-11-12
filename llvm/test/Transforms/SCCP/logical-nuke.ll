; RUN: opt < %s -sccp -S | grep "ret i32 0"

; Test that SCCP has basic knowledge of when and/or nuke overdefined values.

define i32 @test(i32 %X) {
	%Y = and i32 %X, 0		; <i32> [#uses=1]
	ret i32 %Y
}

