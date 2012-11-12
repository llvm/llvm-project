; RUN: %lli -mtriple=%mcjit_triple -use-mcjit %s test

declare i32 @puts(i8*)

define i32 @main(i32 %argc.1, i8** %argv.1) {
	%tmp.5 = getelementptr i8** %argv.1, i64 1		; <i8**> [#uses=1]
	%tmp.6 = load i8** %tmp.5		; <i8*> [#uses=1]
	%tmp.0 = call i32 @puts( i8* %tmp.6 )		; <i32> [#uses=0]
	ret i32 0
}

