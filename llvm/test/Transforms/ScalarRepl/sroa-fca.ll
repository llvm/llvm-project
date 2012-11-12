; RUN: opt < %s -scalarrepl | llvm-dis
; Make sure that SROA "scalar conversion" can handle first class aggregates.

define i64 @test({i32, i32} %A) {
	%X = alloca i64
	%Y = bitcast i64* %X to {i32,i32}*
	store {i32,i32} %A, {i32,i32}* %Y
	
	%Q = load i64* %X
	ret i64 %Q
}

define {i32,i32} @test2(i64 %A) {
	%X = alloca i64
	%Y = bitcast i64* %X to {i32,i32}*
	store i64 %A, i64* %X
	
	%Q = load {i32,i32}* %Y
	ret {i32,i32} %Q
}

