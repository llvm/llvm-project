; Test that GCSE uses basicaa to do alias analysis, which is capable of
; disambiguating some obvious cases.  All loads should be removable in
; this testcase.

; RUN: opt < %s -aa-pipeline=basic-aa -passes=gvn,instcombine,dce -S | FileCheck %s

@A = global i32 7
@B = global i32 8

; CHECK:      define i32 @test()
; CHECK-NEXT:   store i32 123, ptr @B
; CHECK-NEXT:   ret i32 0

define i32 @test() {
	%A1 = load i32, ptr @A

	store i32 123, ptr @B  ; Store cannot alias @A

	%A2 = load i32, ptr @A
	%X = sub i32 %A1, %A2
	ret i32 %X
}

; CHECK:      define i32 @test2()
; CHECK-NEXT:   br label %Loop
; CHECK:      Loop:
; CHECK-NEXT:   store i32 0, ptr @B
; CHECK-NEXT:   br i1 true, label %out, label %Loop
; CHECK:      out:
; CHECK-NEXT:   ret i32 0

define i32 @test2() {
        %A1 = load i32, ptr @A
        br label %Loop
Loop:
        %AP = phi i32 [0, %0], [%X, %Loop]
        store i32 %AP, ptr @B  ; Store cannot alias @A

        %A2 = load i32, ptr @A
        %X = sub i32 %A1, %A2
        %c = icmp eq i32 %X, 0
        br i1 %c, label %out, label %Loop

out:
        ret i32 %X
}

declare void @external()

; CHECK:      define i32 @test3()
; CHECK-NEXT:   call void @external()
; CHECK-NEXT:   ret i32 7

define i32 @test3() {
	%X = alloca i32
	store i32 7, ptr %X
	call void @external()
	%V = load i32, ptr %X
	ret i32 %V
}

