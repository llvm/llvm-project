; RUN: %lli -jit-kind=mcjit %s > /dev/null
; RUN: %lli %s > /dev/null

define i32 @bar(ptr %X) {
        ; pointer should be 4 byte aligned!
	%P = alloca double		; <ptr> [#uses=1]
	%R = ptrtoint ptr %P to i32		; <i32> [#uses=1]
	%A = and i32 %R, 3		; <i32> [#uses=1]
	ret i32 %A
}

define i32 @main() {
	%SP = alloca i8		; <ptr> [#uses=1]
	%X = add i32 0, 0		; <i32> [#uses=1]
	alloca i8, i32 %X		; <ptr>:1 [#uses=0]
	call i32 @bar( ptr %SP )		; <i32>:2 [#uses=1]
	ret i32 %2
}
