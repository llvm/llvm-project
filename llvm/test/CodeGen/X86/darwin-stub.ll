; RUN: llvm-as < %s | llc -mtriple=i386-apple-darwin  |     grep stub
; RUN: llvm-as < %s | llc -mtriple=i386-apple-darwin9 | not grep stub

@"\01LC" = internal constant [13 x i8] c"Hello World!\00"		; <[13 x i8]*> [#uses=1]

define i32 @main() nounwind {
entry:
	%0 = tail call i32 @puts(i8* getelementptr ([13 x i8]* @"\01LC", i32 0, i32 0)) nounwind		; <i32> [#uses=0]
	ret i32 0
}

declare i32 @puts(i8*)
