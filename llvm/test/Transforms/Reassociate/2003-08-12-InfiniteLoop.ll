; RUN: opt < %s -passes=reassociate -disable-output

define i32 @test(i32 %A.1, i32 %B.1, i32 %C.1, i32 %D.1) {
	%tmp.16 = and i32 %A.1, %B.1		; <i32> [#uses=1]
	%tmp.18 = and i32 %tmp.16, %C.1		; <i32> [#uses=1]
	%tmp.20 = and i32 %tmp.18, %D.1		; <i32> [#uses=1]
	ret i32 %tmp.20
}

