; Make sure this testcase is supported by all code generators
; RUN: llc < %s

declare i64 @llvm.ctpop.i64(i64)

declare i32 @llvm.ctpop.i32(i32)

declare i16 @llvm.ctpop.i16(i16)

declare i8 @llvm.ctpop.i8(i8)

define void @ctpoptest(i8 %A, i16 %B, i32 %C, i64 %D, ptr %AP, ptr %BP, ptr %CP, ptr %DP) {
	%a = call i8 @llvm.ctpop.i8( i8 %A )		; <i8> [#uses=1]
	%b = call i16 @llvm.ctpop.i16( i16 %B )		; <i16> [#uses=1]
	%c = call i32 @llvm.ctpop.i32( i32 %C )		; <i32> [#uses=1]
	%d = call i64 @llvm.ctpop.i64( i64 %D )		; <i64> [#uses=1]
	store i8 %a, ptr %AP
	store i16 %b, ptr %BP
	store i32 %c, ptr %CP
	store i64 %d, ptr %DP
	ret void
}

declare i64 @llvm.ctlz.i64(i64, i1)

declare i32 @llvm.ctlz.i32(i32, i1)

declare i16 @llvm.ctlz.i16(i16, i1)

declare i8 @llvm.ctlz.i8(i8, i1)

define void @ctlztest(i8 %A, i16 %B, i32 %C, i64 %D, ptr %AP, ptr %BP, ptr %CP, ptr %DP) {
	%a = call i8 @llvm.ctlz.i8( i8 %A, i1 true )		; <i8> [#uses=1]
	%b = call i16 @llvm.ctlz.i16( i16 %B, i1 true )		; <i16> [#uses=1]
	%c = call i32 @llvm.ctlz.i32( i32 %C, i1 true )		; <i32> [#uses=1]
	%d = call i64 @llvm.ctlz.i64( i64 %D, i1 true )		; <i64> [#uses=1]
	store i8 %a, ptr %AP
	store i16 %b, ptr %BP
	store i32 %c, ptr %CP
	store i64 %d, ptr %DP
	ret void
}

declare i64 @llvm.cttz.i64(i64, i1)

declare i32 @llvm.cttz.i32(i32, i1)

declare i16 @llvm.cttz.i16(i16, i1)

declare i8 @llvm.cttz.i8(i8, i1)

define void @cttztest(i8 %A, i16 %B, i32 %C, i64 %D, ptr %AP, ptr %BP, ptr %CP, ptr %DP) {
	%a = call i8 @llvm.cttz.i8( i8 %A, i1 true )		; <i8> [#uses=1]
	%b = call i16 @llvm.cttz.i16( i16 %B, i1 true )		; <i16> [#uses=1]
	%c = call i32 @llvm.cttz.i32( i32 %C, i1 true )		; <i32> [#uses=1]
	%d = call i64 @llvm.cttz.i64( i64 %D, i1 true )		; <i64> [#uses=1]
	store i8 %a, ptr %AP
	store i16 %b, ptr %BP
	store i32 %c, ptr %CP
	store i64 %d, ptr %DP
	ret void
}
