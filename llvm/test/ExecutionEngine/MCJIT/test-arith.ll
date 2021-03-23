; RUN: %lli %s > /dev/null

define i32 @main() {
	%A = add i8 0, 12		; <i8> [#uses=1]
	%B = sub i8 %A, 1		; <i8> [#uses=2]
	%C = mul i8 %B, %B		; <i8> [#uses=2]
	%D = sdiv i8 %C, %C		; <i8> [#uses=2]
	%E = srem i8 %D, %D		; <i8> [#uses=0]
	%F = udiv i8 5, 6		; <i8> [#uses=0]
	%G = urem i8 6, 5		; <i8> [#uses=0]
	%A.upgrd.1 = add i16 0, 12		; <i16> [#uses=1]
	%B.upgrd.2 = sub i16 %A.upgrd.1, 1		; <i16> [#uses=2]
	%C.upgrd.3 = mul i16 %B.upgrd.2, %B.upgrd.2		; <i16> [#uses=2]
	%D.upgrd.4 = sdiv i16 %C.upgrd.3, %C.upgrd.3		; <i16> [#uses=2]
	%E.upgrd.5 = srem i16 %D.upgrd.4, %D.upgrd.4		; <i16> [#uses=0]
	%F.upgrd.6 = udiv i16 5, 6		; <i16> [#uses=0]
	%G.upgrd.7 = urem i32 6, 5		; <i32> [#uses=0]
	%A.upgrd.8 = add i32 0, 12		; <i32> [#uses=1]
	%B.upgrd.9 = sub i32 %A.upgrd.8, 1		; <i32> [#uses=2]
	%C.upgrd.10 = mul i32 %B.upgrd.9, %B.upgrd.9		; <i32> [#uses=2]
	%D.upgrd.11 = sdiv i32 %C.upgrd.10, %C.upgrd.10		; <i32> [#uses=2]
	%E.upgrd.12 = srem i32 %D.upgrd.11, %D.upgrd.11		; <i32> [#uses=0]
	%F.upgrd.13 = udiv i32 5, 6		; <i32> [#uses=0]
	%G1 = urem i32 6, 5		; <i32> [#uses=0]
	%A.upgrd.14 = add i64 0, 12		; <i64> [#uses=1]
	%B.upgrd.15 = sub i64 %A.upgrd.14, 1		; <i64> [#uses=2]
	%C.upgrd.16 = mul i64 %B.upgrd.15, %B.upgrd.15		; <i64> [#uses=2]
	%D.upgrd.17 = sdiv i64 %C.upgrd.16, %C.upgrd.16		; <i64> [#uses=2]
	%E.upgrd.18 = srem i64 %D.upgrd.17, %D.upgrd.17		; <i64> [#uses=0]
	%F.upgrd.19 = udiv i64 5, 6		; <i64> [#uses=0]
	%G.upgrd.20 = urem i64 6, 5		; <i64> [#uses=0]
	ret i32 0
}

