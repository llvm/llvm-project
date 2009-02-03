; RUN: llvm-as < %s | opt -tailduplicate -taildup-threshold=2 | llvm-dis | grep add | not grep uses=1

define i32 @test1(i1 %C, i32 %A, i32* %P) {
entry:
	br i1 %C, label %L1, label %L2
L1:		; preds = %entry
	store i32 1, i32* %P
	br label %L2
L2:		; preds = %L1, %entry
	%X = add i32 %A, 17		; <i32> [#uses=1]
	ret i32 %X
}

define i32 @test2(i1 %C, i32 %A, i32* %P) {
entry:
	br i1 %C, label %L1, label %L2
L1:		; preds = %entry
	store i32 1, i32* %P
	br label %L3
L2:		; preds = %entry
	store i32 7, i32* %P
	br label %L3
L3:		; preds = %L2, %L1
	%X = add i32 %A, 17		; <i32> [#uses=1]
	ret i32 %X
}

