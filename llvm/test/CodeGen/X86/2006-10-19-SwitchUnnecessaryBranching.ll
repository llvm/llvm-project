; RUN: llvm-as < %s | llc -march=x86 | %prcontext je 1 | \
; RUN:   grep BB1_1:

@str = internal constant [14 x i8] c"Hello world!\0A\00"		; <[14 x i8]*> [#uses=1]
@str.upgrd.1 = internal constant [13 x i8] c"Blah world!\0A\00"		; <[13 x i8]*> [#uses=1]

define i32 @main(i32 %argc, i8** %argv) {
entry:
	switch i32 %argc, label %UnifiedReturnBlock [
		 i32 1, label %bb
		 i32 2, label %bb2
	]

bb:		; preds = %entry
	%tmp1 = tail call i32 (i8*, ...)* @printf( i8* getelementptr ([14 x i8]* @str, i32 0, i64 0) )		; <i32> [#uses=0]
	ret i32 0

bb2:		; preds = %entry
	%tmp4 = tail call i32 (i8*, ...)* @printf( i8* getelementptr ([13 x i8]* @str.upgrd.1, i32 0, i64 0) )		; <i32> [#uses=0]
	ret i32 0

UnifiedReturnBlock:		; preds = %entry
	ret i32 0
}

declare i32 @printf(i8*, ...)
