; RUN: opt < %s -passes=instcombine -disable-output

define <2 x i32> @f() {
	ret <2 x i32> undef
}

define i32 @g() {
	%x = call i32 @f( )		; <i32> [#uses=1]
	ret i32 %x
}
