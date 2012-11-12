; RUN: %lli -mtriple=%mcjit_triple -use-mcjit %s > /dev/null

@A = global i32 0		; <i32*> [#uses=1]

define i32 @main() {
	%Ret = call i32 @test( i1 true, i32 0 )		; <i32> [#uses=1]
	ret i32 %Ret
}

define i32 @test(i1 %c, i32 %A) {
	br i1 %c, label %Taken1, label %NotTaken
Cont:		; preds = %Taken1, %NotTaken
	%V = phi i32 [ 0, %NotTaken ], [ sub (i32 ptrtoint (i32* @A to i32), i32 1234), %Taken1 ]		; <i32> [#uses=0]
	ret i32 0
NotTaken:		; preds = %0
	br label %Cont
Taken1:		; preds = %0
	%B = icmp eq i32 %A, 0		; <i1> [#uses=1]
	br i1 %B, label %Cont, label %ExitError
ExitError:		; preds = %Taken1
	ret i32 12
}

