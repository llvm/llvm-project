; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep bitcast

define void @f(i16 %y) {
	ret void
}

define i32 @g(i32 %y) {
	%x = call i32 bitcast (void (i16)* @f to i32 (i32)*)( i32 %y )		; <i32> [#uses=1]
	ret i32 %x
}
