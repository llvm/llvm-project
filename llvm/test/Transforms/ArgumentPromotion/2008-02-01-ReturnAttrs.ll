; RUN: opt < %s -argpromotion -S | grep nounwind | count 2

define internal i32 @deref(i32* %x) nounwind {
entry:
	%tmp2 = load i32* %x, align 4		; <i32> [#uses=1]
	ret i32 %tmp2
}

define i32 @f(i32 %x) {
entry:
	%x_addr = alloca i32		; <i32*> [#uses=2]
	store i32 %x, i32* %x_addr, align 4
	%tmp1 = call i32 @deref( i32* %x_addr ) nounwind 		; <i32> [#uses=1]
	ret i32 %tmp1
}
