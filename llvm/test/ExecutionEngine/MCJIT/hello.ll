; RUN: %lli -mtriple=%mcjit_triple -use-mcjit %s > /dev/null

@.LC0 = internal global [12 x i8] c"Hello World\00"		; <[12 x i8]*> [#uses=1]

declare i32 @puts(i8*)

define i32 @main() {
	%reg210 = call i32 @puts( i8* getelementptr ([12 x i8]* @.LC0, i64 0, i64 0) )		; <i32> [#uses=0]
	ret i32 0
}

