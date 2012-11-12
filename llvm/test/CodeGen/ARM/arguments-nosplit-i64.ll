; RUN: llc < %s -mtriple=arm-linux-gnueabi | not grep r3
; PR4058

define i32 @f(i64 %z, i32 %a, i64 %b) {
	%tmp = call i32 @g(i64 %b)
	ret i32 %tmp
}

declare i32 @g(i64)
