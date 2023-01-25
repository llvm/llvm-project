; RUN: %lli -jit-kind=mcjit %s test
; RUN: %lli %s test

declare i32 @puts(ptr)

define i32 @main(i32 %argc.1, ptr %argv.1) {
	%tmp.5 = getelementptr ptr, ptr %argv.1, i64 1		; <ptr> [#uses=1]
	%tmp.6 = load ptr, ptr %tmp.5		; <ptr> [#uses=1]
	%tmp.0 = call i32 @puts( ptr %tmp.6 )		; <i32> [#uses=0]
	ret i32 0
}

