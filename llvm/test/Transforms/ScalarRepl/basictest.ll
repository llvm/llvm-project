; RUN: opt < %s -scalarrepl -S | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"

define i32 @test1() {
	%X = alloca { i32, float }		; <{ i32, float }*> [#uses=1]
	%Y = getelementptr { i32, float }* %X, i64 0, i32 0		; <i32*> [#uses=2]
	store i32 0, i32* %Y
	%Z = load i32* %Y		; <i32> [#uses=1]
	ret i32 %Z
; CHECK: @test1
; CHECK-NOT: alloca
; CHECK: ret i32 0
}

; PR8980
define i64 @test2(i64 %X) {
	%A = alloca [8 x i8]
        %B = bitcast [8 x i8]* %A to i64*
        
	store i64 %X, i64* %B
        br label %L2
        
L2:
	%Z = load i64* %B		; <i32> [#uses=1]
	ret i64 %Z
; CHECK: @test2
; CHECK-NOT: alloca
; CHECK: ret i64 %X
}

