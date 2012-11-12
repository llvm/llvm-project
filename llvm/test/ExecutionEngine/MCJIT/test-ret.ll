; RUN: %lli -mtriple=%mcjit_triple -use-mcjit %s > /dev/null

; test return instructions
define void @test1() {
	ret void
}

define i8 @test2() {
	ret i8 1
}

define i8 @test3() {
	ret i8 1
}

define i16 @test4() {
	ret i16 -1
}

define i16 @test5() {
	ret i16 -1
}

define i32 @main() {
	ret i32 0
}

define i32 @test6() {
	ret i32 4
}

define i64 @test7() {
	ret i64 0
}

define i64 @test8() {
	ret i64 0
}

define float @test9() {
	ret float 1.000000e+00
}

define double @test10() {
	ret double 2.000000e+00
}
