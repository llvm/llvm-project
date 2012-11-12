; RUN: opt < %s -basicaa -functionattrs -S | grep readonly | count 2

define i32 @f() {
entry:
	%tmp = call i32 @e( )		; <i32> [#uses=1]
	ret i32 %tmp
}

declare i32 @e() readonly
