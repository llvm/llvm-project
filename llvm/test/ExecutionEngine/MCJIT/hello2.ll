; RUN: %lli %s > /dev/null

@X = global i32 7		; <i32*> [#uses=0]
@msg = internal global [13 x i8] c"Hello World\0A\00"		; <[13 x i8]*> [#uses=1]

declare void @printf([13 x i8]*, ...)

define void @bar() {
	call void ([13 x i8]*, ...) @printf( [13 x i8]* @msg )
	ret void
}

define i32 @main() {
	call void @bar( )
	ret i32 0
}

