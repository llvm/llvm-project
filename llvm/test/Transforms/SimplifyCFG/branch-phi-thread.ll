; RUN: opt < %s -simplifycfg -adce -S | \
; RUN:   not grep "call void @f1"
; END.

declare void @f1()

declare void @f2()

declare void @f3()

declare void @f4()

define i32 @test1(i32 %X, i1 %D) {
E:
	%C = icmp eq i32 %X, 0		; <i1> [#uses=2]
	br i1 %C, label %T, label %F
T:		; preds = %A, %E
	br i1 %C, label %B, label %A
A:		; preds = %T
	call void @f1( )
	br i1 %D, label %T, label %F
B:		; preds = %T
	call void @f2( )
	ret i32 345
F:		; preds = %A, %E
	call void @f3( )
	ret i32 123
}

define i32 @test2(i32 %X, i1 %D) {
E:
	%C = icmp eq i32 %X, 0		; <i1> [#uses=2]
	br i1 %C, label %T, label %F
T:		; preds = %A, %E
	%P = phi i1 [ true, %E ], [ %C, %A ]		; <i1> [#uses=1]
	br i1 %P, label %B, label %A
A:		; preds = %T
	call void @f1( )
	br i1 %D, label %T, label %F
B:		; preds = %T
	call void @f2( )
	ret i32 345
F:		; preds = %A, %E
	call void @f3( )
	ret i32 123
}

define i32 @test3(i32 %X, i1 %D, i32* %AP, i32* %BP) {
E:
	%C = icmp eq i32 %X, 0		; <i1> [#uses=2]
	br i1 %C, label %T, label %F
T:		; preds = %A, %E
	call void @f3( )
	%XX = load i32* %AP		; <i32> [#uses=1]
	store i32 %XX, i32* %BP
	br i1 %C, label %B, label %A
A:		; preds = %T
	call void @f1( )
	br i1 %D, label %T, label %F
B:		; preds = %T
	call void @f2( )
	ret i32 345
F:		; preds = %A, %E
	call void @f3( )
	ret i32 123
}
