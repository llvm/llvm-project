; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-apple-darwin9.5
; rdar://6499616

@"\01LC" = internal constant [13 x i8] c"conftest.val\00"		; <[13 x i8]*> [#uses=1]

define i32 @main() nounwind {
entry:
	%0 = call i8* @fopen(i8* getelementptr ([13 x i8], [13 x i8]* @"\01LC", i32 0, i32 0), i8* null) nounwind		; <i8*> [#uses=0]
	unreachable
}

declare i8* @fopen(i8*, i8*)
