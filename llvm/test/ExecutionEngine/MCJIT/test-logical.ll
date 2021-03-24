; RUN: %lli %s > /dev/null

define i32 @main() {
	%A = and i8 4, 8		; <i8> [#uses=2]
	%B = or i8 %A, 7		; <i8> [#uses=1]
	%C = xor i8 %B, %A		; <i8> [#uses=0]
	%A.upgrd.1 = and i16 4, 8		; <i16> [#uses=2]
	%B.upgrd.2 = or i16 %A.upgrd.1, 7		; <i16> [#uses=1]
	%C.upgrd.3 = xor i16 %B.upgrd.2, %A.upgrd.1		; <i16> [#uses=0]
	%A.upgrd.4 = and i32 4, 8		; <i32> [#uses=2]
	%B.upgrd.5 = or i32 %A.upgrd.4, 7		; <i32> [#uses=1]
	%C.upgrd.6 = xor i32 %B.upgrd.5, %A.upgrd.4		; <i32> [#uses=0]
	%A.upgrd.7 = and i64 4, 8		; <i64> [#uses=2]
	%B.upgrd.8 = or i64 %A.upgrd.7, 7		; <i64> [#uses=1]
	%C.upgrd.9 = xor i64 %B.upgrd.8, %A.upgrd.7		; <i64> [#uses=0]
	ret i32 0
}

