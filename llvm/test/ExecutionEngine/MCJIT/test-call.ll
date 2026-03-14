; RUN: %lli -jit-kind=mcjit %s > /dev/null
; RUN: %lli %s > /dev/null

declare void @exit(i32)

define i32 @test(i8 %C, i16 %S) {
	%X = trunc i16 %S to i8		; <i8> [#uses=1]
	%Y = zext i8 %X to i32		; <i32> [#uses=1]
	ret i32 %Y
}

define void @FP(ptr %F) {
	%X = call i32 @test( i8 123, i16 1024 )		; <i32> [#uses=1]
	call void %F( i32 %X )
	ret void
}

define i32 @main() {
	call void @FP( ptr @exit )
	ret i32 1
}

