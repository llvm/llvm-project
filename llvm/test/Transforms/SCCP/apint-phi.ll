; RUN: opt < %s -sccp -S | not grep phi

define i999 @test(i999%A, i1 %c) {
bb1:
	br label %BB2
BB2:
	%V = phi i999 [2, %bb1], [%A, %BB4]
	br label %BB3

BB3:
        %E = trunc i999 %V to i1
        %F = and i1 %E, %c
	br i1 %F, label %BB4, label %BB5
BB4:
	br label %BB2

BB5:
	ret i999 %V
}
