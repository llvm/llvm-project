; RUN: opt < %s -scalarrepl -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32"
target triple = "i386-pc-linux-gnu"

%nested = type { i32, [4 x i32] }

; Check that a GEP with a non-zero first index does not prevent SROA as long
; as the resulting offset corresponds to an element in the alloca.
define i32 @test1() {
; CHECK: @test1
; CHECK-NOT: = i160
; CHECK: ret i32 undef
	%A = alloca %nested
	%B = getelementptr %nested* %A, i32 0, i32 1, i32 0
	%C = getelementptr i32* %B, i32 2
	%D = load i32* %C
	ret i32 %D
}

; But, if the offset is out of range, then it should not be transformed.
define i32 @test2() {
; CHECK: @test2
; CHECK: i160
	%A = alloca %nested
	%B = getelementptr %nested* %A, i32 0, i32 1, i32 0
	%C = getelementptr i32* %B, i32 4
	%D = load i32* %C
	ret i32 %D
}

; Try it with a bitcast and single GEP....
define i32 @test3() {
; CHECK: @test3
; CHECK-NOT: = i160
; CHECK: ret i32 undef
	%A = alloca %nested
	%B = bitcast %nested* %A to i32*
	%C = getelementptr i32* %B, i32 2
	%D = load i32* %C
	ret i32 %D
}

; ...and again make sure that out-of-range accesses are not transformed.
define i32 @test4() {
; CHECK: @test4
; CHECK: i160
	%A = alloca %nested
	%B = bitcast %nested* %A to i32*
	%C = getelementptr i32* %B, i32 -1
	%D = load i32* %C
	ret i32 %D
}
