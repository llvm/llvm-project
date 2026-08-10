; RUN: %lli -jit-kind=mcjit %s > /dev/null
; RUN: %lli %s > /dev/null

@X = global i32 7		; <ptr> [#uses=0]
@msg = internal global [13 x i8] c"Hello World\0A\00"		; <ptr> [#uses=1]

declare void @printf(ptr, ...)

define void @bar() {
	call void (ptr, ...) @printf( ptr @msg )
	ret void
}

define i32 @main() {
	call void @bar( )
	ret i32 0
}

