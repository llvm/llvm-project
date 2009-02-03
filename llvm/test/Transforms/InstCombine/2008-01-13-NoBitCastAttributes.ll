; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep bitcast | count 2

define i32 @b(i32* inreg  %x) signext  {
	ret i32 0
}

define void @c(...) {
	ret void
}

define void @g(i32* %y) {
	call i32 bitcast (i32 (i32*)* @b to i32 (i32)*)( i32 zeroext  0 )		; <i32>:2 [#uses=0]
	call void bitcast (void (...)* @c to void (i32*)*)( i32* sret  null )
	ret void
}
