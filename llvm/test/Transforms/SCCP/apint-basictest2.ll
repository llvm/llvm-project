; This is a basic correctness check for constant propagation.  The add
; instruction and phi instruction should be eliminated.

; RUN: opt < %s -passes=sccp -S | not grep phi
; RUN: opt < %s -passes=sccp -S | not grep add

define i128 @test(i1 %B) {
	br i1 %B, label %BB1, label %BB2
BB1:
	%Val = add i128 0, 1
	br label %BB3
BB2:
	br label %BB3
BB3:
	%Ret = phi i128 [%Val, %BB1], [1, %BB2]
	ret i128 %Ret
}
