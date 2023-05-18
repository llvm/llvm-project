; In this test, a local alloca cannot alias an incoming argument.

; RUN: opt < %s -aa-pipeline=basic-aa -passes=gvn,instcombine -S | FileCheck %s

; CHECK:      define i32 @test
; CHECK-NEXT: ret i32 0

define i32 @test(ptr %P) {
	%X = alloca i32
	%V1 = load i32, ptr %P
	store i32 0, ptr %X
	%V2 = load i32, ptr %P
	%Diff = sub i32 %V1, %V2
	ret i32 %Diff
}
