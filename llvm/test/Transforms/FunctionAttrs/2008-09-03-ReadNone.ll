; RUN: llvm-as < %s | opt -functionattrs | llvm-dis | grep readnone | count 4
@x = global i32 0

declare i32 @e() readnone

define i32 @f() {
	%tmp = call i32 @e( )		; <i32> [#uses=1]
	ret i32 %tmp
}

define i32 @g() readonly {
	ret i32 0
}

define i32 @h() readnone {
	%tmp = load i32* @x		; <i32> [#uses=1]
	ret i32 %tmp
}
