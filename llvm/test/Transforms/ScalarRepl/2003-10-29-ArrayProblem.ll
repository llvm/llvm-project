; RUN: opt < %s -scalarrepl -S | grep "alloca %%T"

%T = type { [80 x i8], i32, i32 }
declare i32 @.callback_1(i8*)

declare void @.iter_2(i32 (i8*)*, i8*)

define i32 @main() {
	%d = alloca %T		; <{ [80 x i8], i32, i32 }*> [#uses=2]
	%tmp.0 = getelementptr %T* %d, i64 0, i32 2		; <i32*> [#uses=1]
	store i32 0, i32* %tmp.0
	%tmp.1 = getelementptr %T* %d, i64 0, i32 0, i64 0		; <i8*> [#uses=1]
	call void @.iter_2( i32 (i8*)* @.callback_1, i8* %tmp.1 )
	ret i32 0
}

