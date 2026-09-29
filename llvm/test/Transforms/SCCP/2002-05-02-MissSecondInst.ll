; RUN: opt < %s -passes=sccp -S | not grep sub

define void @test3(i32, i32) {
	add i32 0, 0		; <i32>:3 [#uses=0]
	sub i32 0, 4		; <i32>:4 [#uses=0]
	ret void
}

