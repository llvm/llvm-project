; Test that the ToAsciiOptimizer works correctly
; RUN: llvm-as < %s | opt -simplify-libcalls | llvm-dis | \
; RUN:   not grep {call.*toascii}

declare i32 @toascii(i32)

define i32 @main() {
	%val1 = call i32 @toascii( i32 1 )		; <i32> [#uses=1]
	%val2 = call i32 @toascii( i32 0 )		; <i32> [#uses=1]
	%val3 = call i32 @toascii( i32 127 )		; <i32> [#uses=1]
	%val4 = call i32 @toascii( i32 128 )		; <i32> [#uses=1]
	%val5 = call i32 @toascii( i32 255 )		; <i32> [#uses=1]
	%val6 = call i32 @toascii( i32 256 )		; <i32> [#uses=1]
	%rslt1 = add i32 %val1, %val2		; <i32> [#uses=1]
	%rslt2 = add i32 %val3, %val4		; <i32> [#uses=1]
	%rslt3 = add i32 %val5, %val6		; <i32> [#uses=1]
	%rslt4 = add i32 %rslt1, %rslt2		; <i32> [#uses=1]
	%rslt5 = add i32 %rslt4, %rslt3		; <i32> [#uses=1]
	ret i32 %rslt5
}

