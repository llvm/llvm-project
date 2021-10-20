; Mem2reg should not insert dead PHI nodes!  The naive algorithm inserts a PHI
;  node in L3, even though there is no load of %A in anything dominated by L3.

; RUN: opt < %s -passes=mem2reg -S | not grep phi

define void @test(i32 %B, i1 %C) {
	%A = alloca i32		; <i32*> [#uses=4]
	store i32 %B, i32* %A
	br i1 %C, label %L1, label %L2
L1:		; preds = %0
	store i32 %B, i32* %A
	%D = load i32, i32* %A		; <i32> [#uses=1]
	call void @test( i32 %D, i1 false )
	br label %L3
L2:		; preds = %0
	%E = load i32, i32* %A		; <i32> [#uses=1]
	call void @test( i32 %E, i1 true )
	br label %L3
L3:		; preds = %L2, %L1
	ret void
}

