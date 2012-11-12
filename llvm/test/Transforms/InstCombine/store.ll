; RUN: opt < %s -instcombine -S | FileCheck %s

define void @test1(i32* %P) {
        store i32 undef, i32* %P
        store i32 123, i32* undef
        store i32 124, i32* null
        ret void
; CHECK: @test1(
; CHECK-NEXT: store i32 123, i32* undef
; CHECK-NEXT: store i32 undef, i32* null
; CHECK-NEXT: ret void
}

define void @test2(i32* %P) {
        %X = load i32* %P               ; <i32> [#uses=1]
        %Y = add i32 %X, 0              ; <i32> [#uses=1]
        store i32 %Y, i32* %P
        ret void
; CHECK: @test2
; CHECK-NEXT: ret void
}

;; Simple sinking tests

; "if then else"
define i32 @test3(i1 %C) {
	%A = alloca i32
        br i1 %C, label %Cond, label %Cond2

Cond:
        store i32 -987654321, i32* %A
        br label %Cont

Cond2:
	store i32 47, i32* %A
	br label %Cont

Cont:
	%V = load i32* %A
	ret i32 %V
; CHECK: @test3
; CHECK-NOT: alloca
; CHECK: Cont:
; CHECK-NEXT:  %storemerge = phi i32 [ 47, %Cond2 ], [ -987654321, %Cond ]
; CHECK-NEXT:  ret i32 %storemerge
}

; "if then"
define i32 @test4(i1 %C) {
	%A = alloca i32
	store i32 47, i32* %A
        br i1 %C, label %Cond, label %Cont

Cond:
        store i32 -987654321, i32* %A
        br label %Cont

Cont:
	%V = load i32* %A
	ret i32 %V
; CHECK: @test4
; CHECK-NOT: alloca
; CHECK: Cont:
; CHECK-NEXT:  %storemerge = phi i32 [ -987654321, %Cond ], [ 47, %0 ]
; CHECK-NEXT:  ret i32 %storemerge
}

; "if then"
define void @test5(i1 %C, i32* %P) {
	store i32 47, i32* %P, align 1
        br i1 %C, label %Cond, label %Cont

Cond:
        store i32 -987654321, i32* %P, align 1
        br label %Cont

Cont:
	ret void
; CHECK: @test5
; CHECK: Cont:
; CHECK-NEXT:  %storemerge = phi i32
; CHECK-NEXT:  store i32 %storemerge, i32* %P, align 1
; CHECK-NEXT:  ret void
}

