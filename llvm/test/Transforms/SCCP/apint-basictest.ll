; This is a basic sanity check for constant propagation.  The add instruction
; should be eliminated.

; RUN: opt < %s -sccp -S | not grep add

define i128 @test(i1 %B) {
	br i1 %B, label %BB1, label %BB2
BB1:
	%Val = add i128 0, 1
	br label %BB3
BB2:
	br label %BB3
BB3:
	%Ret = phi i128 [%Val, %BB1], [2, %BB2]
	ret i128 %Ret
}
