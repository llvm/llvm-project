; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32--
; RUN: llc -verify-machineinstrs < %s -mtriple=ppc64--
; PR1399

@.str = internal constant [13 x i8] c"Hello World!\00"

define i32 @main() {
	%tmp2 = tail call i32 @puts( i8* getelementptr ([13 x i8], [13 x i8]* @.str, i32 0, i64 0) )
	ret i32 0
}

declare i32 @puts(i8*)
