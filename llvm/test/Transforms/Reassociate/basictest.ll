; RUN: opt < %s -reassociate -gvn -instcombine -S | FileCheck %s
; RUN: opt < %s -passes='reassociate,gvn,instcombine' -S | FileCheck %s

define i32 @test1(i32 %arg) {
  %tmp1 = sub i32 -12, %arg
  %tmp2 = add i32 %tmp1, 12
  ret i32 %tmp2

; CHECK-LABEL: @test1
; CHECK-NEXT: sub i32 0, %arg
; CHECK-NEXT: ret i32
}

define i32 @test2(i32 %reg109, i32 %reg1111) {
  %reg115 = add i32 %reg109, -30
  %reg116 = add i32 %reg115, %reg1111
  %reg117 = add i32 %reg116, 30
  ret i32 %reg117

; CHECK-LABEL: @test2
; CHECK-NEXT: %reg117 = add i32 %reg1111, %reg109
; CHECK-NEXT: ret i32 %reg117
}

@e = external global i32
@a = external global i32
@b = external global i32
@c = external global i32
@f = external global i32

define void @test3() {
  %A = load i32, i32* @a
  %B = load i32, i32* @b
  %C = load i32, i32* @c
  %t1 = add i32 %A, %B
  %t2 = add i32 %t1, %C
  %t3 = add i32 %C, %A
  %t4 = add i32 %t3, %B
  ; e = (a+b)+c;
  store i32 %t2, i32* @e
  ; f = (a+c)+b
  store i32 %t4, i32* @f
  ret void

; CHECK-LABEL: @test3
; CHECK: add i32
; CHECK: add i32
; CHECK-NOT: add i32
; CHECK: ret void
}

define void @test4() {
  %A = load i32, i32* @a
  %B = load i32, i32* @b
  %C = load i32, i32* @c
  %t1 = add i32 %A, %B
  %t2 = add i32 %t1, %C
  %t3 = add i32 %C, %A
  %t4 = add i32 %t3, %B
  ; e = c+(a+b)
  store i32 %t2, i32* @e
  ; f = (c+a)+b
  store i32 %t4, i32* @f
  ret void

; CHECK-LABEL: @test4
; CHECK: add i32
; CHECK: add i32
; CHECK-NOT: add i32
; CHECK: ret void
}

define void @test5() {
  %A = load i32, i32* @a
  %B = load i32, i32* @b
  %C = load i32, i32* @c
  %t1 = add i32 %B, %A
  %t2 = add i32 %t1, %C
  %t3 = add i32 %C, %A
  %t4 = add i32 %t3, %B
  ; e = c+(b+a)
  store i32 %t2, i32* @e
  ; f = (c+a)+b
  store i32 %t4, i32* @f
  ret void

; CHECK-LABEL: @test5
; CHECK: add i32
; CHECK: add i32
; CHECK-NOT: add i32
; CHECK: ret void
}

define i32 @test6() {
  %tmp.0 = load i32, i32* @a
  %tmp.1 = load i32, i32* @b
  ; (a+b)
  %tmp.2 = add i32 %tmp.0, %tmp.1
  %tmp.4 = load i32, i32* @c
  ; (a+b)+c
  %tmp.5 = add i32 %tmp.2, %tmp.4
  ; (a+c)
  %tmp.8 = add i32 %tmp.0, %tmp.4
  ; (a+c)+b
  %tmp.11 = add i32 %tmp.8, %tmp.1
  ; X ^ X = 0
  %RV = xor i32 %tmp.5, %tmp.11
  ret i32 %RV

; CHECK-LABEL: @test6
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

; CHECK-LABEL: @test7
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

; CHECK-LABEL: @test8
; CHECK-NEXT: %A = mul i32 %Y, %X
; CHECK-NEXT: %C = sub i32 %Z, %A
; CHECK-NEXT: ret i32 %C
}

; PR5458
define i32 @test9(i32 %X) {
  %Y = mul i32 %X, 47
  %Z = add i32 %Y, %Y
  ret i32 %Z
; CHECK-LABEL: @test9
; CHECK-NEXT: mul i32 %X, 94
; CHECK-NEXT: ret i32
}

define i32 @test10(i32 %X) {
  %Y = add i32 %X ,%X
  %Z = add i32 %Y, %X
  ret i32 %Z
; CHECK-LABEL: @test10
; CHECK-NEXT: mul i32 %X, 3
; CHECK-NEXT: ret i32
}

define i32 @test11(i32 %W) {
  %X = mul i32 %W, 127
  %Y = add i32 %X ,%X
  %Z = add i32 %Y, %X
  ret i32 %Z
; CHECK-LABEL: @test11
; CHECK-NEXT: mul i32 %W, 381
; CHECK-NEXT: ret i32
}

declare void @mumble(i32)

define i32 @test12(i32 %X) {
  %X.neg = sub nsw nuw i32 0, %X
  call void @mumble(i32 %X.neg)
  %A = sub i32 1, %X
  %B = sub i32 2, %X
  %C = sub i32 3, %X
  %Y = add i32 %A ,%B
  %Z = add i32 %Y, %C
  ret i32 %Z
; CHECK-LABEL: @test12
; CHECK: %[[mul:.*]] = mul i32 %X, -3
; CHECK-NEXT: add i32 %[[mul]], 6
; CHECK-NEXT: ret i32
}

define i32 @test13(i32 %X1, i32 %X2, i32 %X3) {
  %A = sub i32 0, %X1
  %B = mul i32 %A, %X2   ; -X1*X2
  %C = mul i32 %X1, %X3  ; X1*X3
  %D = add i32 %B, %C    ; -X1*X2 + X1*X3 -> X1*(X3-X2)
  ret i32 %D
; CHECK-LABEL: @test13
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

; CHECK-LABEL: @test14
; CHECK-NEXT: %[[SUB:.*]] = sub i32 %X1, %X2
; CHECK-NEXT: mul i32 %[[SUB]], 47
; CHECK-NEXT: ret i32
}

; Do not reassociate expressions of type i1
define i32 @test15(i32 %X1, i32 %X2, i32 %X3) {
  %A = icmp ne i32 %X1, 0
  %B = icmp slt i32 %X2, %X3
  %C = and i1 %A, %B
  %D = select i1 %C, i32 %X1, i32 0
  ret i32 %D
; CHECK-LABEL: @test15
; CHECK: and i1 %A, %B
}

; PR30256 - previously this asserted.
; CHECK-LABEL: @test16
; CHECK: %[[FACTOR:.*]] = mul i64 %a, -4
; CHECK-NEXT: %[[RES:.*]] = add i64 %[[FACTOR]], %b
; CHECK-NEXT: ret i64 %[[RES]]
define i64 @test16(i1 %cmp, i64 %a, i64 %b) {
entry:
  %shl = shl i64 %a, 1
  %shl.neg = sub i64 0, %shl
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %add1 = add i64 %shl.neg, %shl.neg
  %add2 = add i64 %add1, %b
  ret i64 %add2

if.end:                                           ; preds = %entry
  ret i64 0
}

; CHECK-LABEL: @test17
; CHECK: %[[A:.*]] = mul i32 %X4, %X3
; CHECK-NEXT:  %[[C:.*]] = mul i32 %[[A]], %X1
; CHECK-NEXT: %[[D:.*]] = mul i32 %[[A]], %X2
; CHECK-NEXT: %[[E:.*]] = xor i32 %[[C]], %[[D]]
; CHECK-NEXT: ret i32 %[[E]]
define i32 @test17(i32 %X1, i32 %X2, i32 %X3, i32 %X4) {
  %A = mul i32 %X3, %X1
  %B = mul i32 %X3, %X2
  %C = mul i32 %A, %X4
  %D = mul i32 %B, %X4
  %E = xor i32 %C, %D
  ret i32 %E
}
