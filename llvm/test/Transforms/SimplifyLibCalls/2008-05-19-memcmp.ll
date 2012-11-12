; RUN: opt < %s -simplify-libcalls -S | grep i32
; PR2341

@_2E_str = external constant [5 x i8]		; <[5 x i8]*> [#uses=1]

declare i32 @memcmp(i8*, i8*, i32) nounwind readonly 

define i1 @f(i8** %start_addr) {
entry:
	%tmp4 = load i8** %start_addr, align 4		; <i8*> [#uses=1]
	%tmp5 = call i32 @memcmp( i8* %tmp4, i8* getelementptr ([5 x i8]* @_2E_str, i32 0, i32 0), i32 4 ) nounwind readonly 		; <i32> [#uses=1]
	%tmp6 = icmp eq i32 %tmp5, 0		; <i1> [#uses=1]
	ret i1 %tmp6
}
