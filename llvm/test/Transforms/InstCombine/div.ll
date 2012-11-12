; This test makes sure that div instructions are properly eliminated.

; RUN: opt < %s -instcombine -S | FileCheck %s

define i32 @test1(i32 %A) {
        %B = sdiv i32 %A, 1             ; <i32> [#uses=1]
        ret i32 %B
; CHECK: @test1
; CHECK-NEXT: ret i32 %A
}

define i32 @test2(i32 %A) {
        ; => Shift
        %B = udiv i32 %A, 8             ; <i32> [#uses=1]
        ret i32 %B
; CHECK: @test2
; CHECK-NEXT: lshr i32 %A, 3
}

define i32 @test3(i32 %A) {
        ; => 0, don't need to keep traps
        %B = sdiv i32 0, %A             ; <i32> [#uses=1]
        ret i32 %B
; CHECK: @test3
; CHECK-NEXT: ret i32 0
}

define i32 @test4(i32 %A) {
        ; 0-A
        %B = sdiv i32 %A, -1            ; <i32> [#uses=1]
        ret i32 %B
; CHECK: @test4
; CHECK-NEXT: sub i32 0, %A
}

define i32 @test5(i32 %A) {
        %B = udiv i32 %A, -16           ; <i32> [#uses=1]
        %C = udiv i32 %B, -4            ; <i32> [#uses=1]
        ret i32 %C
; CHECK: @test5
; CHECK-NEXT: ret i32 0
}

define i1 @test6(i32 %A) {
        %B = udiv i32 %A, 123           ; <i32> [#uses=1]
        ; A < 123
        %C = icmp eq i32 %B, 0          ; <i1> [#uses=1]
        ret i1 %C
; CHECK: @test6
; CHECK-NEXT: icmp ult i32 %A, 123
}

define i1 @test7(i32 %A) {
        %B = udiv i32 %A, 10            ; <i32> [#uses=1]
        ; A >= 20 && A < 30
        %C = icmp eq i32 %B, 2          ; <i1> [#uses=1]
        ret i1 %C
; CHECK: @test7
; CHECK-NEXT: add i32 %A, -20
; CHECK-NEXT: icmp ult i32
}

define i1 @test8(i8 %A) {
        %B = udiv i8 %A, 123            ; <i8> [#uses=1]
        ; A >= 246
        %C = icmp eq i8 %B, 2           ; <i1> [#uses=1]
        ret i1 %C
; CHECK: @test8
; CHECK-NEXT: icmp ugt i8 %A, -11
}

define i1 @test9(i8 %A) {
        %B = udiv i8 %A, 123            ; <i8> [#uses=1]
        ; A < 246
        %C = icmp ne i8 %B, 2           ; <i1> [#uses=1]
        ret i1 %C
; CHECK: @test9
; CHECK-NEXT: icmp ult i8 %A, -10
}

define i32 @test10(i32 %X, i1 %C) {
        %V = select i1 %C, i32 64, i32 8                ; <i32> [#uses=1]
        %R = udiv i32 %X, %V            ; <i32> [#uses=1]
        ret i32 %R
; CHECK: @test10
; CHECK-NEXT: select i1 %C, i32 6, i32 3
; CHECK-NEXT: lshr i32 %X
}

define i32 @test11(i32 %X, i1 %C) {
        %A = select i1 %C, i32 1024, i32 32             ; <i32> [#uses=1]
        %B = udiv i32 %X, %A            ; <i32> [#uses=1]
        ret i32 %B
; CHECK: @test11
; CHECK-NEXT: select i1 %C, i32 10, i32 5
; CHECK-NEXT: lshr i32 %X
}

; PR2328
define i32 @test12(i32 %x) nounwind  {
	%tmp3 = udiv i32 %x, %x		; 1
	ret i32 %tmp3
; CHECK: @test12
; CHECK-NEXT: ret i32 1
}

define i32 @test13(i32 %x) nounwind  {
	%tmp3 = sdiv i32 %x, %x		; 1
	ret i32 %tmp3
; CHECK: @test13
; CHECK-NEXT: ret i32 1
}

define i32 @test14(i8 %x) nounwind {
	%zext = zext i8 %x to i32
	%div = udiv i32 %zext, 257	; 0
	ret i32 %div
; CHECK: @test14
; CHECK-NEXT: ret i32 0
}

; PR9814
define i32 @test15(i32 %a, i32 %b) nounwind {
  %shl = shl i32 1, %b
  %div = lshr i32 %shl, 2
  %div2 = udiv i32 %a, %div
  ret i32 %div2
; CHECK: @test15
; CHECK-NEXT: add i32 %b, -2
; CHECK-NEXT: lshr i32 %a, 
; CHECK-NEXT: ret i32
}


