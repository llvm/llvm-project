; Test that the IsDigitOptimizer works correctly
; RUN: llvm-as < %s | opt -simplify-libcalls | llvm-dis | \
; RUN:   not grep call

declare i32 @isdigit(i32)

declare i32 @isascii(i32)

define i32 @main() {
	%val1 = call i32 @isdigit( i32 47 )		; <i32> [#uses=1]
	%val2 = call i32 @isdigit( i32 48 )		; <i32> [#uses=1]
	%val3 = call i32 @isdigit( i32 57 )		; <i32> [#uses=1]
	%val4 = call i32 @isdigit( i32 58 )		; <i32> [#uses=1]
	%rslt1 = add i32 %val1, %val2		; <i32> [#uses=1]
	%rslt2 = add i32 %val3, %val4		; <i32> [#uses=1]
	%sum = add i32 %rslt1, %rslt2		; <i32> [#uses=1]
	%rslt = call i32 @isdigit( i32 %sum )		; <i32> [#uses=1]
	%tmp = call i32 @isascii( i32 %rslt )		; <i32> [#uses=1]
	ret i32 %tmp
}

