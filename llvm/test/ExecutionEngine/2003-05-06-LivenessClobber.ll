; This testcase should return with an exit code of 1.
;
; RUN: not %lli %s
; XFAIL: arm

@test = global i64 0		; <i64*> [#uses=1]

define internal i64 @test.upgrd.1() {
	%tmp.0 = load i64* @test		; <i64> [#uses=1]
	%tmp.1 = add i64 %tmp.0, 1		; <i64> [#uses=1]
	ret i64 %tmp.1
}

define i32 @main() {
	%L = call i64 @test.upgrd.1( )		; <i64> [#uses=1]
	%I = trunc i64 %L to i32		; <i32> [#uses=1]
	ret i32 %I
}


