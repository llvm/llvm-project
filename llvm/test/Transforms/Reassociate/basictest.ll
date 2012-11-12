; With reassociation, constant folding can eliminate the 12 and -12 constants.
;
; RUN: opt < %s -reassociate  -gvn -instcombine -S | FileCheck %s

define i32 @test1(i32 %arg) {
	%tmp1 = sub i32 -12, %arg
	%tmp2 = add i32 %tmp1, 12
	ret i32 %tmp2
; CHECK: @test1
; CHECK-NEXT: sub i32 0, %arg
; CHECK-NEXT: ret i32
}

define i32 @test2(i32 %reg109, i32 %reg1111) {
	%reg115 = add i32 %reg109, -30		; <i32> [#uses=1]
	%reg116 = add i32 %reg115, %reg1111		; <i32> [#uses=1]
	%reg117 = add i32 %reg116, 30		; <i32> [#uses=1]
	ret i32 %reg117
; CHECK: @test2
; CHECK-NEXT: add i32 %reg1111, %reg109
; CHECK-NEXT: ret i32
}

@e = external global i32		; <i32*> [#uses=3]
@a = external global i32		; <i32*> [#uses=3]
@b = external global i32		; <i32*> [#uses=3]
@c = external global i32		; <i32*> [#uses=3]
@f = external global i32		; <i32*> [#uses=3]

define void @test3() {
	%A = load i32* @a		; <i32> [#uses=2]
	%B = load i32* @b		; <i32> [#uses=2]
	%C = load i32* @c		; <i32> [#uses=2]
	%t1 = add i32 %A, %B		; <i32> [#uses=1]
	%t2 = add i32 %t1, %C		; <i32> [#uses=1]
	%t3 = add i32 %C, %A		; <i32> [#uses=1]
	%t4 = add i32 %t3, %B		; <i32> [#uses=1]
	; e = (a+b)+c;
        store i32 %t2, i32* @e
        ; f = (a+c)+b
	store i32 %t4, i32* @f
	ret void
; CHECK: @test3
; CHECK: add i32
; CHECK: add i32
; CHECK-NOT: add i32
; CHECK: ret void
}

define void @test4() {
	%A = load i32* @a		; <i32> [#uses=2]
	%B = load i32* @b		; <i32> [#uses=2]
	%C = load i32* @c		; <i32> [#uses=2]
	%t1 = add i32 %A, %B		; <i32> [#uses=1]
	%t2 = add i32 %t1, %C		; <i32> [#uses=1]
	%t3 = add i32 %C, %A		; <i32> [#uses=1]
	%t4 = add i32 %t3, %B		; <i32> [#uses=1]
	; e = c+(a+b)
        store i32 %t2, i32* @e
        ; f = (c+a)+b
	store i32 %t4, i32* @f
	ret void
; CHECK: @test4
; CHECK: add i32
; CHECK: add i32
; CHECK-NOT: add i32
; CHECK: ret void
}

define void @test5() {
	%A = load i32* @a		; <i32> [#uses=2]
	%B = load i32* @b		; <i32> [#uses=2]
	%C = load i32* @c		; <i32> [#uses=2]
	%t1 = add i32 %B, %A		; <i32> [#uses=1]
	%t2 = add i32 %t1, %C		; <i32> [#uses=1]
	%t3 = add i32 %C, %A		; <i32> [#uses=1]
	%t4 = add i32 %t3, %B		; <i32> [#uses=1]
	; e = c+(b+a)
        store i32 %t2, i32* @e
        ; f = (c+a)+b
	store i32 %t4, i32* @f
	ret void
; CHECK: @test5
; CHECK: add i32
; CHECK: add i32
; CHECK-NOT: add i32
; CHECK: ret void
}

define i32 @test6() {
	%tmp.0 = load i32* @a
	%tmp.1 = load i32* @b
        ; (a+b)
	%tmp.2 = add i32 %tmp.0, %tmp.1
	%tmp.4 = load i32* @c
	; (a+b)+c
        %tmp.5 = add i32 %tmp.2, %tmp.4
	; (a+c)
        %tmp.8 = add i32 %tmp.0, %tmp.4
	; (a+c)+b
        %tmp.11 = add i32 %tmp.8, %tmp.1
	; X ^ X = 0
        %RV = xor i32 %tmp.5, %tmp.11
	ret i32 %RV
; CHECK: @test6
; CHECK: ret i32 0
}

; This should be one add and two multiplies.
define i32 @test7(i32 %A, i32 %B, i32 %C) {
 ; A*A*B + A*C*A
	%aa = mul i32 %A, %A
	%aab = mul i32 %aa, %B
	%ac = mul i32 %A, %C
	%aac = mul i32 %ac, %A
	%r = add i32 %aab, %aac
	ret i32 %r
; CHECK: @test7
; CHECK-NEXT: add i32 %C, %B
; CHECK-NEXT: mul i32 
; CHECK-NEXT: mul i32 
; CHECK-NEXT: ret i32 
}


define i32 @test8(i32 %X, i32 %Y, i32 %Z) {
	%A = sub i32 0, %X
	%B = mul i32 %A, %Y
        ; (-X)*Y + Z -> Z-X*Y
	%C = add i32 %B, %Z
	ret i32 %C
; CHECK: @test8
; CHECK-NEXT: %A = mul i32 %Y, %X
; CHECK-NEXT: %C = sub i32 %Z, %A
; CHECK-NEXT: ret i32 %C
}


; PR5458
define i32 @test9(i32 %X) {
  %Y = mul i32 %X, 47
  %Z = add i32 %Y, %Y
  ret i32 %Z
; CHECK: @test9
; CHECK-NEXT: mul i32 %X, 94
; CHECK-NEXT: ret i32
}

define i32 @test10(i32 %X) {
  %Y = add i32 %X ,%X
  %Z = add i32 %Y, %X
  ret i32 %Z
; CHECK: @test10
; CHECK-NEXT: mul i32 %X, 3
; CHECK-NEXT: ret i32
}

define i32 @test11(i32 %W) {
  %X = mul i32 %W, 127
  %Y = add i32 %X ,%X
  %Z = add i32 %Y, %X
  ret i32 %Z
; CHECK: @test11
; CHECK-NEXT: mul i32 %W, 381
; CHECK-NEXT: ret i32
}

define i32 @test12(i32 %X) {
  %A = sub i32 1, %X
  %B = sub i32 2, %X
  %C = sub i32 3, %X

  %Y = add i32 %A ,%B
  %Z = add i32 %Y, %C
  ret i32 %Z
; CHECK: @test12
; CHECK-NEXT: mul i32 %X, -3
; CHECK-NEXT: add i32{{.*}}, 6
; CHECK-NEXT: ret i32
}

define i32 @test13(i32 %X1, i32 %X2, i32 %X3) {
  %A = sub i32 0, %X1
  %B = mul i32 %A, %X2   ; -X1*X2
  %C = mul i32 %X1, %X3  ; X1*X3
  %D = add i32 %B, %C    ; -X1*X2 + X1*X3 -> X1*(X3-X2)
  ret i32 %D
; CHECK: @test13
; CHECK-NEXT: sub i32 %X3, %X2
; CHECK-NEXT: mul i32 {{.*}}, %X1
; CHECK-NEXT: ret i32
}

; PR5359
define i32 @test14(i32 %X1, i32 %X2) {
  %B = mul i32 %X1, 47   ; X1*47
  %C = mul i32 %X2, -47  ; X2*-47
  %D = add i32 %B, %C    ; X1*47 + X2*-47 -> 47*(X1-X2)
  ret i32 %D
; CHECK: @test14
; CHECK-NEXT: sub i32 %X1, %X2
; CHECK-NEXT: mul i32 {{.*}}, 47
; CHECK-NEXT: ret i32
}

; Do not reassociate expressions of type i1
define i32 @test15(i32 %X1, i32 %X2, i32 %X3) {
  %A = icmp ne i32 %X1, 0
  %B = icmp slt i32 %X2, %X3
  %C = and i1 %A, %B
  %D = select i1 %C, i32 %X1, i32 0
  ret i32 %D
; CHECK: @test15
; CHECK: and i1 %A, %B
}

