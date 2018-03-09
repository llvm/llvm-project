; RUN: llc < %s -mtriple=i686-- -enable-legalize-types-checking
; PR3393

define void @foo(i32 inreg %x) {
	%t709 = select i1 false, i32 0, i32 %x		; <i32> [#uses=1]
	%t711 = add i32 %t709, 1		; <i32> [#uses=4]
	%t801 = icmp slt i32 %t711, 0		; <i1> [#uses=1]
	%t712 = zext i32 %t711 to i64		; <i64> [#uses=1]
	%t804 = select i1 %t801, i64 0, i64 %t712		; <i64> [#uses=1]
	store i64 %t804, i64* null
	%t815 = icmp slt i32 %t711, 0		; <i1> [#uses=1]
	%t814 = sext i32 %t711 to i64		; <i64> [#uses=1]
	%t816 = select i1 %t815, i64 0, i64 %t814		; <i64> [#uses=1]
	store i64 %t816, i64* null
	unreachable
}
