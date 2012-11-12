; RUN: %lli %s > /dev/null
; XFAIL: arm

define i32 @bar(i8* %X) {
        ; pointer should be 4 byte aligned!
	%P = alloca double		; <double*> [#uses=1]
	%R = ptrtoint double* %P to i32		; <i32> [#uses=1]
	%A = and i32 %R, 3		; <i32> [#uses=1]
	ret i32 %A
}

define i32 @main() {
	%SP = alloca i8		; <i8*> [#uses=1]
	%X = add i32 0, 0		; <i32> [#uses=1]
	alloca i8, i32 %X		; <i8*>:1 [#uses=0]
	call i32 @bar( i8* %SP )		; <i32>:2 [#uses=1]
	ret i32 %2
}
