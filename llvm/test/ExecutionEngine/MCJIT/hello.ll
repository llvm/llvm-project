; RUN: %lli -jit-kind=mcjit %s > /dev/null
; RUN: %lli %s > /dev/null

@.LC0 = internal global [12 x i8] c"Hello World\00"		; <ptr> [#uses=1]

declare i32 @puts(ptr)

define i32 @main() {
	%reg210 = call i32 @puts( ptr @.LC0 )		; <i32> [#uses=0]
	ret i32 0
}

