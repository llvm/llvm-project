; RUN: opt < %s -instcombine -S | FileCheck %s

; PR1822

target datalayout = "e-p:64:64-p1:16:16-p2:32:32:32-p3:64:64:64"

define i32 @test1(i32 %A, i32 %B) {
        %C = select i1 false, i32 %A, i32 %B
        ret i32 %C
; CHECK-LABEL: @test1(
; CHECK: ret i32 %B
}

define i32 @test2(i32 %A, i32 %B) {
        %C = select i1 true, i32 %A, i32 %B
        ret i32 %C
; CHECK-LABEL: @test2(
; CHECK: ret i32 %A
}


define i32 @test3(i1 %C, i32 %I) {
        ; V = I
        %V = select i1 %C, i32 %I, i32 %I
        ret i32 %V
; CHECK-LABEL: @test3(
; CHECK: ret i32 %I
}

define i1 @test4(i1 %C) {
        ; V = C
        %V = select i1 %C, i1 true, i1 false
        ret i1 %V
; CHECK-LABEL: @test4(
; CHECK: ret i1 %C
}

define i1 @test5(i1 %C) {
        ; V = !C
        %V = select i1 %C, i1 false, i1 true
        ret i1 %V
; CHECK-LABEL: @test5(
; CHECK: xor i1 %C, true
; CHECK: ret i1
}

define i32 @test6(i1 %C) {
        ; V = cast C to int
        %V = select i1 %C, i32 1, i32 0
        ret i32 %V
; CHECK-LABEL: @test6(
; CHECK: %V = zext i1 %C to i32
; CHECK: ret i32 %V
}

define i1 @test7(i1 %C, i1 %X) {
; CHECK-LABEL: @test7(
; CHECK-NEXT:    [[R:%.*]] = or i1 %C, %X
; CHECK-NEXT:    ret i1 [[R]]
;
  %R = select i1 %C, i1 true, i1 %X
  ret i1 %R
}

define <2 x i1> @test7vec(<2 x i1> %C, <2 x i1> %X) {
; CHECK-LABEL: @test7vec(
; CHECK-NEXT:    [[R:%.*]] = or <2 x i1> %C, %X
; CHECK-NEXT:    ret <2 x i1> [[R]]
;
  %R = select <2 x i1> %C, <2 x i1> <i1 true, i1 true>, <2 x i1> %X
  ret <2 x i1> %R
}

define i1 @test8(i1 %C, i1 %X) {
; CHECK-LABEL: @test8(
; CHECK-NEXT:    [[R:%.*]] = and i1 %C, %X
; CHECK-NEXT:    ret i1 [[R]]
;
  %R = select i1 %C, i1 %X, i1 false
  ret i1 %R
}

define <2 x i1> @test8vec(<2 x i1> %C, <2 x i1> %X) {
; CHECK-LABEL: @test8vec(
; CHECK-NEXT:    [[R:%.*]] = and <2 x i1> %C, %X
; CHECK-NEXT:    ret <2 x i1> [[R]]
;
  %R = select <2 x i1> %C, <2 x i1> %X, <2 x i1> <i1 false, i1 false>
  ret <2 x i1> %R
}

define i1 @test9(i1 %C, i1 %X) {
; CHECK-LABEL: @test9(
; CHECK-NEXT:    [[NOT_C:%.*]] = xor i1 %C, true
; CHECK-NEXT:    [[R:%.*]] = and i1 [[NOT_C]], %X
; CHECK-NEXT:    ret i1 [[R]]
;
  %R = select i1 %C, i1 false, i1 %X
  ret i1 %R
}

define <2 x i1> @test9vec(<2 x i1> %C, <2 x i1> %X) {
; CHECK-LABEL: @test9vec(
; CHECK-NEXT:    [[NOT_C:%.*]] = xor <2 x i1> %C, <i1 true, i1 true>
; CHECK-NEXT:    [[R:%.*]] = and <2 x i1> [[NOT_C]], %X
; CHECK-NEXT:    ret <2 x i1> [[R]]
;
  %R = select <2 x i1> %C, <2 x i1> <i1 false, i1 false>, <2 x i1> %X
  ret <2 x i1> %R
}

define i1 @test10(i1 %C, i1 %X) {
; CHECK-LABEL: @test10(
; CHECK-NEXT:    [[NOT_C:%.*]] = xor i1 %C, true
; CHECK-NEXT:    [[R:%.*]] = or i1 [[NOT_C]], %X
; CHECK-NEXT:    ret i1 [[R]]
;
  %R = select i1 %C, i1 %X, i1 true
  ret i1 %R
}

define <2 x i1> @test10vec(<2 x i1> %C, <2 x i1> %X) {
; CHECK-LABEL: @test10vec(
; CHECK-NEXT:    [[NOT_C:%.*]] = xor <2 x i1> %C, <i1 true, i1 true>
; CHECK-NEXT:    [[R:%.*]] = or <2 x i1> [[NOT_C]], %X
; CHECK-NEXT:    ret <2 x i1> [[R]]
;
  %R = select <2 x i1> %C, <2 x i1> %X, <2 x i1> <i1 true, i1 true>
  ret <2 x i1> %R
}

define i1 @test23(i1 %a, i1 %b) {
; CHECK-LABEL: @test23(
; CHECK-NEXT:    [[C:%.*]] = and i1 %a, %b
; CHECK-NEXT:    ret i1 [[C]]
;
  %c = select i1 %a, i1 %b, i1 %a
  ret i1 %c
}

define <2 x i1> @test23vec(<2 x i1> %a, <2 x i1> %b) {
; CHECK-LABEL: @test23vec(
; CHECK-NEXT:    [[C:%.*]] = and <2 x i1> %a, %b
; CHECK-NEXT:    ret <2 x i1> [[C]]
;
  %c = select <2 x i1> %a, <2 x i1> %b, <2 x i1> %a
  ret <2 x i1> %c
}

define i1 @test24(i1 %a, i1 %b) {
; CHECK-LABEL: @test24(
; CHECK-NEXT:    [[C:%.*]] = or i1 %a, %b
; CHECK-NEXT:    ret i1 [[C]]
;
  %c = select i1 %a, i1 %a, i1 %b
  ret i1 %c
}

define <2 x i1> @test24vec(<2 x i1> %a, <2 x i1> %b) {
; CHECK-LABEL: @test24vec(
; CHECK-NEXT:    [[C:%.*]] = or <2 x i1> %a, %b
; CHECK-NEXT:    ret <2 x i1> [[C]]
;
  %c = select <2 x i1> %a, <2 x i1> %a, <2 x i1> %b
  ret <2 x i1> %c
}

define i1 @test62(i1 %A, i1 %B) {
; CHECK-LABEL: @test62(
; CHECK-NEXT:    [[NOT:%.*]] = xor i1 %A, true
; CHECK-NEXT:    [[C:%.*]] = and i1 [[NOT]], %B
; CHECK-NEXT:    ret i1 [[C]]
;
  %not = xor i1 %A, true
  %C = select i1 %A, i1 %not, i1 %B
  ret i1 %C
}

define <2 x i1> @test62vec(<2 x i1> %A, <2 x i1> %B) {
; CHECK-LABEL: @test62vec(
; CHECK-NEXT:    [[NOT:%.*]] = xor <2 x i1> %A, <i1 true, i1 true>
; CHECK-NEXT:    [[C:%.*]] = and <2 x i1> [[NOT]], %B
; CHECK-NEXT:    ret <2 x i1> [[C]]
;
  %not = xor <2 x i1> %A, <i1 true, i1 true>
  %C = select <2 x i1> %A, <2 x i1> %not, <2 x i1> %B
  ret <2 x i1> %C
}

define i1 @test63(i1 %A, i1 %B) {
; CHECK-LABEL: @test63(
; CHECK-NEXT:    [[NOT:%.*]] = xor i1 %A, true
; CHECK-NEXT:    [[C:%.*]] = or i1 [[NOT]], %B
; CHECK-NEXT:    ret i1 [[C]]
;
  %not = xor i1 %A, true
  %C = select i1 %A, i1 %B, i1 %not
  ret i1 %C
}

define <2 x i1> @test63vec(<2 x i1> %A, <2 x i1> %B) {
; CHECK-LABEL: @test63vec(
; CHECK-NEXT:    [[NOT:%.*]] = xor <2 x i1> %A, <i1 true, i1 true>
; CHECK-NEXT:    [[C:%.*]] = or <2 x i1> [[NOT]], %B
; CHECK-NEXT:    ret <2 x i1> [[C]]
;
  %not = xor <2 x i1> %A, <i1 true, i1 true>
  %C = select <2 x i1> %A, <2 x i1> %B, <2 x i1> %not
  ret <2 x i1> %C
}

define i32 @test11(i32 %a) {
        %C = icmp eq i32 %a, 0
        %R = select i1 %C, i32 0, i32 1
        ret i32 %R
; CHECK-LABEL: @test11(
; CHECK: icmp ne i32 %a, 0
; CHECK: %R = zext i1
; CHECK: ret i32 %R
}

define i32 @test12(i1 %cond, i32 %a) {
        %b = or i32 %a, 1
        %c = select i1 %cond, i32 %b, i32 %a
        ret i32 %c
; CHECK-LABEL: @test12(
; CHECK: %b = zext i1 %cond to i32
; CHECK: %c = or i32 %b, %a
; CHECK: ret i32 %c
}

define i32 @test12a(i1 %cond, i32 %a) {
        %b = ashr i32 %a, 1
        %c = select i1 %cond, i32 %b, i32 %a
        ret i32 %c
; CHECK-LABEL: @test12a(
; CHECK: %b = zext i1 %cond to i32
; CHECK: %c = ashr i32 %a, %b
; CHECK: ret i32 %c
}

define i32 @test12b(i1 %cond, i32 %a) {
; CHECK-LABEL: @test12b(
; CHECK-NEXT:    [[NOT_COND:%.*]] = xor i1 %cond, true
; CHECK-NEXT:    [[B:%.*]] = zext i1 [[NOT_COND]] to i32
; CHECK-NEXT:    [[D:%.*]] = ashr i32 %a, [[B]]
; CHECK-NEXT:    ret i32 [[D]]
;
  %b = ashr i32 %a, 1
  %d = select i1 %cond, i32 %a, i32 %b
  ret i32 %d
}

define i32 @test13(i32 %a, i32 %b) {
        %C = icmp eq i32 %a, %b
        %V = select i1 %C, i32 %a, i32 %b
        ret i32 %V
; CHECK-LABEL: @test13(
; CHECK: ret i32 %b
}

define i32 @test13a(i32 %a, i32 %b) {
        %C = icmp ne i32 %a, %b
        %V = select i1 %C, i32 %a, i32 %b
        ret i32 %V
; CHECK-LABEL: @test13a(
; CHECK: ret i32 %a
}

define i32 @test13b(i32 %a, i32 %b) {
        %C = icmp eq i32 %a, %b
        %V = select i1 %C, i32 %b, i32 %a
        ret i32 %V
; CHECK-LABEL: @test13b(
; CHECK: ret i32 %a
}

define i1 @test14a(i1 %C, i32 %X) {
        %V = select i1 %C, i32 %X, i32 0
        ; (X < 1) | !C
        %R = icmp slt i32 %V, 1
        ret i1 %R
; CHECK-LABEL: @test14a(
; CHECK: icmp slt i32 %X, 1
; CHECK: xor i1 %C, true
; CHECK: or i1
; CHECK: ret i1 %R
}

define i1 @test14b(i1 %C, i32 %X) {
        %V = select i1 %C, i32 0, i32 %X
        ; (X < 1) | C
        %R = icmp slt i32 %V, 1
        ret i1 %R
; CHECK-LABEL: @test14b(
; CHECK: icmp slt i32 %X, 1
; CHECK: or i1
; CHECK: ret i1 %R
}

;; Code sequence for (X & 16) ? 16 : 0
define i32 @test15a(i32 %X) {
        %t1 = and i32 %X, 16
        %t2 = icmp eq i32 %t1, 0
        %t3 = select i1 %t2, i32 0, i32 16
        ret i32 %t3
; CHECK-LABEL: @test15a(
; CHECK: %t1 = and i32 %X, 16
; CHECK: ret i32 %t1
}

;; Code sequence for (X & 32) ? 0 : 24
define i32 @test15b(i32 %X) {
        %t1 = and i32 %X, 32
        %t2 = icmp eq i32 %t1, 0
        %t3 = select i1 %t2, i32 32, i32 0
        ret i32 %t3
; CHECK-LABEL: @test15b(
; CHECK: %t1 = and i32 %X, 32
; CHECK: xor i32 %t1, 32
; CHECK: ret i32
}

;; Alternate code sequence for (X & 16) ? 16 : 0
define i32 @test15c(i32 %X) {
        %t1 = and i32 %X, 16
        %t2 = icmp eq i32 %t1, 16
        %t3 = select i1 %t2, i32 16, i32 0
        ret i32 %t3
; CHECK-LABEL: @test15c(
; CHECK: %t1 = and i32 %X, 16
; CHECK: ret i32 %t1
}

;; Alternate code sequence for (X & 16) ? 16 : 0
define i32 @test15d(i32 %X) {
        %t1 = and i32 %X, 16
        %t2 = icmp ne i32 %t1, 0
        %t3 = select i1 %t2, i32 16, i32 0
        ret i32 %t3
; CHECK-LABEL: @test15d(
; CHECK: %t1 = and i32 %X, 16
; CHECK: ret i32 %t1
}

;; (a & 128) ? 256 : 0
define i32 @test15e(i32 %X) {
        %t1 = and i32 %X, 128
        %t2 = icmp ne i32 %t1, 0
        %t3 = select i1 %t2, i32 256, i32 0
        ret i32 %t3
; CHECK-LABEL: @test15e(
; CHECK: %t1 = shl i32 %X, 1
; CHECK: and i32 %t1, 256
; CHECK: ret i32
}

;; (a & 128) ? 0 : 256
define i32 @test15f(i32 %X) {
        %t1 = and i32 %X, 128
        %t2 = icmp ne i32 %t1, 0
        %t3 = select i1 %t2, i32 0, i32 256
        ret i32 %t3
; CHECK-LABEL: @test15f(
; CHECK: %t1 = shl i32 %X, 1
; CHECK: and i32 %t1, 256
; CHECK: xor i32 %{{.*}}, 256
; CHECK: ret i32
}

;; (a & 8) ? -1 : -9
define i32 @test15g(i32 %X) {
        %t1 = and i32 %X, 8
        %t2 = icmp ne i32 %t1, 0
        %t3 = select i1 %t2, i32 -1, i32 -9
        ret i32 %t3
; CHECK-LABEL: @test15g(
; CHECK-NEXT: %1 = or i32 %X, -9
; CHECK-NEXT: ret i32 %1
}

;; (a & 8) ? -9 : -1
define i32 @test15h(i32 %X) {
        %t1 = and i32 %X, 8
        %t2 = icmp ne i32 %t1, 0
        %t3 = select i1 %t2, i32 -9, i32 -1
        ret i32 %t3
; CHECK-LABEL: @test15h(
; CHECK-NEXT: %1 = or i32 %X, -9
; CHECK-NEXT: %2 = xor i32 %1, 8
; CHECK-NEXT: ret i32 %2
}

;; (a & 2) ? 577 : 1089
define i32 @test15i(i32 %X) {
        %t1 = and i32 %X, 2
        %t2 = icmp ne i32 %t1, 0
        %t3 = select i1 %t2, i32 577, i32 1089
        ret i32 %t3
; CHECK-LABEL: @test15i(
; CHECK-NEXT: %t1 = shl i32 %X, 8
; CHECK-NEXT: %1 = and i32 %t1, 512
; CHECK-NEXT: %2 = xor i32 %1, 512
; CHECK-NEXT: %3 = add nuw nsw i32 %2, 577
; CHECK-NEXT: ret i32 %3
}

;; (a & 2) ? 1089 : 577
define i32 @test15j(i32 %X) {
        %t1 = and i32 %X, 2
        %t2 = icmp ne i32 %t1, 0
        %t3 = select i1 %t2, i32 1089, i32 577
        ret i32 %t3
; CHECK-LABEL: @test15j(
; CHECK-NEXT: %t1 = shl i32 %X, 8
; CHECK-NEXT: %1 = and i32 %t1, 512
; CHECK-NEXT: %2 = add nuw nsw i32 %1, 577
; CHECK-NEXT: ret i32 %2
}

define i32 @test16(i1 %C, i32* %P) {
        %P2 = select i1 %C, i32* %P, i32* null
        %V = load i32, i32* %P2
        ret i32 %V
; CHECK-LABEL: @test16(
; CHECK-NEXT: %V = load i32, i32* %P
; CHECK: ret i32 %V
}

;; It may be legal to load from a null address in a non-zero address space
define i32 @test16_neg(i1 %C, i32 addrspace(1)* %P) {
        %P2 = select i1 %C, i32 addrspace(1)* %P, i32 addrspace(1)* null
        %V = load i32, i32 addrspace(1)* %P2
        ret i32 %V
; CHECK-LABEL: @test16_neg
; CHECK-NEXT: %P2 = select i1 %C, i32 addrspace(1)* %P, i32 addrspace(1)* null
; CHECK-NEXT: %V = load i32, i32 addrspace(1)* %P2
; CHECK: ret i32 %V
}
define i32 @test16_neg2(i1 %C, i32 addrspace(1)* %P) {
        %P2 = select i1 %C, i32 addrspace(1)* null, i32 addrspace(1)* %P
        %V = load i32, i32 addrspace(1)* %P2
        ret i32 %V
; CHECK-LABEL: @test16_neg2
; CHECK-NEXT: %P2 = select i1 %C, i32 addrspace(1)* null, i32 addrspace(1)* %P
; CHECK-NEXT: %V = load i32, i32 addrspace(1)* %P2
; CHECK: ret i32 %V
}

define i1 @test17(i32* %X, i1 %C) {
        %R = select i1 %C, i32* %X, i32* null
        %RV = icmp eq i32* %R, null
        ret i1 %RV
; CHECK-LABEL: @test17(
; CHECK: icmp eq i32* %X, null
; CHECK: xor i1 %C, true
; CHECK: %RV = or i1
; CHECK: ret i1 %RV
}

define i32 @test18(i32 %X, i32 %Y, i1 %C) {
        %R = select i1 %C, i32 %X, i32 0
        %V = sdiv i32 %Y, %R
        ret i32 %V
; CHECK-LABEL: @test18(
; CHECK: %V = sdiv i32 %Y, %X
; CHECK: ret i32 %V
}

define i32 @test19(i32 %x) {
        %tmp = icmp ugt i32 %x, 2147483647
        %retval = select i1 %tmp, i32 -1, i32 0
        ret i32 %retval
; CHECK-LABEL: @test19(
; CHECK-NEXT: ashr i32 %x, 31
; CHECK-NEXT: ret i32
}

define i32 @test20(i32 %x) {
        %tmp = icmp slt i32 %x, 0
        %retval = select i1 %tmp, i32 -1, i32 0
        ret i32 %retval
; CHECK-LABEL: @test20(
; CHECK-NEXT: ashr i32 %x, 31
; CHECK-NEXT: ret i32
}

define i64 @test21(i32 %x) {
        %tmp = icmp slt i32 %x, 0
        %retval = select i1 %tmp, i64 -1, i64 0
        ret i64 %retval
; CHECK-LABEL: @test21(
; CHECK-NEXT: ashr i32 %x, 31
; CHECK-NEXT: sext i32
; CHECK-NEXT: ret i64
}

define i16 @test22(i32 %x) {
        %tmp = icmp slt i32 %x, 0
        %retval = select i1 %tmp, i16 -1, i16 0
        ret i16 %retval
; CHECK-LABEL: @test22(
; CHECK-NEXT: ashr i32 %x, 31
; CHECK-NEXT: trunc i32
; CHECK-NEXT: ret i16
}

define i32 @test25(i1 %c)  {
entry:
  br i1 %c, label %jump, label %ret
jump:
  br label %ret
ret:
  %a = phi i1 [true, %jump], [false, %entry]
  %b = select i1 %a, i32 10, i32 20
  ret i32 %b
; CHECK-LABEL: @test25(
; CHECK: %a = phi i32 [ 10, %jump ], [ 20, %entry ]
; CHECK-NEXT: ret i32 %a
}

define i32 @test26(i1 %cond)  {
entry:
  br i1 %cond, label %jump, label %ret
jump:
  %c = or i1 false, false
  br label %ret
ret:
  %a = phi i1 [true, %entry], [%c, %jump]
  %b = select i1 %a, i32 20, i32 10
  ret i32 %b
; CHECK-LABEL: @test26(
; CHECK: %a = phi i32 [ 20, %entry ], [ 10, %jump ]
; CHECK-NEXT: ret i32 %a
}

define i32 @test27(i1 %c, i32 %A, i32 %B)  {
entry:
  br i1 %c, label %jump, label %ret
jump:
  br label %ret
ret:
  %a = phi i1 [true, %jump], [false, %entry]
  %b = select i1 %a, i32 %A, i32 %B
  ret i32 %b
; CHECK-LABEL: @test27(
; CHECK: %a = phi i32 [ %A, %jump ], [ %B, %entry ]
; CHECK-NEXT: ret i32 %a
}

define i32 @test28(i1 %cond, i32 %A, i32 %B)  {
entry:
  br i1 %cond, label %jump, label %ret
jump:
  br label %ret
ret:
  %c = phi i32 [%A, %jump], [%B, %entry]
  %a = phi i1 [true, %jump], [false, %entry]
  %b = select i1 %a, i32 %A, i32 %c
  ret i32 %b
; CHECK-LABEL: @test28(
; CHECK: %a = phi i32 [ %A, %jump ], [ %B, %entry ]
; CHECK-NEXT: ret i32 %a
}

define i32 @test29(i1 %cond, i32 %A, i32 %B)  {
entry:
  br i1 %cond, label %jump, label %ret
jump:
  br label %ret
ret:
  %c = phi i32 [%A, %jump], [%B, %entry]
  %a = phi i1 [true, %jump], [false, %entry]
  br label %next

next:
  %b = select i1 %a, i32 %A, i32 %c
  ret i32 %b
; CHECK-LABEL: @test29(
; CHECK: %a = phi i32 [ %A, %jump ], [ %B, %entry ]
; CHECK: ret i32 %a
}


; SMAX(SMAX(x, y), x) -> SMAX(x, y)
define i32 @test30(i32 %x, i32 %y) {
  %cmp = icmp sgt i32 %x, %y
  %cond = select i1 %cmp, i32 %x, i32 %y

  %cmp5 = icmp sgt i32 %cond, %x
  %retval = select i1 %cmp5, i32 %cond, i32 %x
  ret i32 %retval
; CHECK-LABEL: @test30(
; CHECK: ret i32 %cond
}

; UMAX(UMAX(x, y), x) -> UMAX(x, y)
define i32 @test31(i32 %x, i32 %y) {
  %cmp = icmp ugt i32 %x, %y
  %cond = select i1 %cmp, i32 %x, i32 %y
  %cmp5 = icmp ugt i32 %cond, %x
  %retval = select i1 %cmp5, i32 %cond, i32 %x
  ret i32 %retval
; CHECK-LABEL: @test31(
; CHECK: ret i32 %cond
}

; SMIN(SMIN(x, y), x) -> SMIN(x, y)
define i32 @test32(i32 %x, i32 %y) {
  %cmp = icmp sgt i32 %x, %y
  %cond = select i1 %cmp, i32 %y, i32 %x
  %cmp5 = icmp sgt i32 %cond, %x
  %retval = select i1 %cmp5, i32 %x, i32 %cond
  ret i32 %retval
; CHECK-LABEL: @test32(
; CHECK: ret i32 %cond
}

; MAX(MIN(x, y), x) -> x
define i32 @test33(i32 %x, i32 %y) {
  %cmp = icmp sgt i32 %x, %y
  %cond = select i1 %cmp, i32 %y, i32 %x
  %cmp5 = icmp sgt i32 %cond, %x
  %retval = select i1 %cmp5, i32 %cond, i32 %x
  ret i32 %retval
; CHECK-LABEL: @test33(
; CHECK: ret i32 %x
}

; MIN(MAX(x, y), x) -> x
define i32 @test34(i32 %x, i32 %y) {
  %cmp = icmp sgt i32 %x, %y
  %cond = select i1 %cmp, i32 %x, i32 %y
  %cmp5 = icmp sgt i32 %cond, %x
  %retval = select i1 %cmp5, i32 %x, i32 %cond
  ret i32 %retval
; CHECK-LABEL: @test34(
; CHECK: ret i32 %x
}

define i32 @test35(i32 %x) {
  %cmp = icmp sge i32 %x, 0
  %cond = select i1 %cmp, i32 60, i32 100
  ret i32 %cond
; CHECK-LABEL: @test35(
; CHECK: ashr i32 %x, 31
; CHECK: and i32 {{.*}}, 40
; CHECK: add nuw nsw i32 {{.*}}, 60
; CHECK: ret
}

define i32 @test36(i32 %x) {
  %cmp = icmp slt i32 %x, 0
  %cond = select i1 %cmp, i32 60, i32 100
  ret i32 %cond
; CHECK-LABEL: @test36(
; CHECK: ashr i32 %x, 31
; CHECK: and i32 {{.*}}, -40
; CHECK: add nsw i32 {{.*}}, 100
; CHECK: ret
}

define i32 @test37(i32 %x) {
  %cmp = icmp sgt i32 %x, -1
  %cond = select i1 %cmp, i32 1, i32 -1
  ret i32 %cond
; CHECK-LABEL: @test37(
; CHECK: ashr i32 %x, 31
; CHECK: or i32 {{.*}}, 1
; CHECK: ret
}

define i1 @test38(i1 %cond) {
  %zero = alloca i32
  %one = alloca i32
  %ptr = select i1 %cond, i32* %zero, i32* %one
  %isnull = icmp eq i32* %ptr, null
  ret i1 %isnull
; CHECK-LABEL: @test38(
; CHECK: ret i1 false
}

define i1 @test39(i1 %cond, double %x) {
  %s = select i1 %cond, double %x, double 0x7FF0000000000000 ; RHS = +infty
  %cmp = fcmp ule double %x, %s
  ret i1 %cmp
; CHECK-LABEL: @test39(
; CHECK: ret i1 true
}

define i1 @test40(i1 %cond) {
  %a = alloca i32
  %b = alloca i32
  %c = alloca i32
  %s = select i1 %cond, i32* %a, i32* %b
  %r = icmp eq i32* %s, %c
  ret i1 %r
; CHECK-LABEL: @test40(
; CHECK: ret i1 false
}

define i32 @test41(i1 %cond, i32 %x, i32 %y) {
  %z = and i32 %x, %y
  %s = select i1 %cond, i32 %y, i32 %z
  %r = and i32 %x, %s
  ret i32 %r
; CHECK-LABEL: @test41(
; CHECK-NEXT: and i32 %x, %y
; CHECK-NEXT: ret i32
}

define i32 @test42(i32 %x, i32 %y) {
  %b = add i32 %y, -1
  %cond = icmp eq i32 %x, 0
  %c = select i1 %cond, i32 %b, i32 %y
  ret i32 %c
; CHECK-LABEL: @test42(
; CHECK-NEXT: %cond = icmp eq i32 %x, 0
; CHECK-NEXT: %b = sext i1 %cond to i32
; CHECK-NEXT: %c = add i32 %b, %y
; CHECK-NEXT: ret i32 %c
}

; PR8994

; This select instruction can't be eliminated because trying to do so would
; change the number of vector elements. This used to assert.
define i48 @test51(<3 x i1> %icmp, <3 x i16> %tmp) {
; CHECK-LABEL: @test51(
  %select = select <3 x i1> %icmp, <3 x i16> zeroinitializer, <3 x i16> %tmp
; CHECK: select <3 x i1>
  %tmp2 = bitcast <3 x i16> %select to i48
  ret i48 %tmp2
}

; Allow select promotion even if there are multiple uses of bitcasted ops.
; Hoisting the selects allows later pattern matching to see that these are min/max ops.

define void @min_max_bitcast(<4 x float> %a, <4 x float> %b, <4 x i32>* %ptr1, <4 x i32>* %ptr2) {
; CHECK-LABEL: @min_max_bitcast(
; CHECK-NEXT:    [[CMP:%.*]] = fcmp olt <4 x float> %a, %b
; CHECK-NEXT:    [[SEL1_V:%.*]] = select <4 x i1> [[CMP]], <4 x float> %a, <4 x float> %b
; CHECK-NEXT:    [[SEL2_V:%.*]] = select <4 x i1> [[CMP]], <4 x float> %b, <4 x float> %a
; CHECK-NEXT:    [[TMP1:%.*]] = bitcast <4 x i32>* %ptr1 to <4 x float>*
; CHECK-NEXT:    store <4 x float> [[SEL1_V]], <4 x float>* [[TMP1]], align 16
; CHECK-NEXT:    [[TMP2:%.*]] = bitcast <4 x i32>* %ptr2 to <4 x float>*
; CHECK-NEXT:    store <4 x float> [[SEL2_V]], <4 x float>* [[TMP2]], align 16
; CHECK-NEXT:    ret void
;
  %cmp = fcmp olt <4 x float> %a, %b
  %bc1 = bitcast <4 x float> %a to <4 x i32>
  %bc2 = bitcast <4 x float> %b to <4 x i32>
  %sel1 = select <4 x i1> %cmp, <4 x i32> %bc1, <4 x i32> %bc2
  %sel2 = select <4 x i1> %cmp, <4 x i32> %bc2, <4 x i32> %bc1
  store <4 x i32> %sel1, <4 x i32>* %ptr1
  store <4 x i32> %sel2, <4 x i32>* %ptr2
  ret void
}

; To avoid potential backend problems, we don't do the same transform for other casts.

define void @truncs_before_selects(<4 x float> %f1, <4 x float> %f2, <4 x i64> %a, <4 x i64> %b, <4 x i32>* %ptr1, <4 x i32>* %ptr2) {
; CHECK-LABEL: @truncs_before_selects(
; CHECK-NEXT:    [[CMP:%.*]] = fcmp olt <4 x float> %f1, %f2
; CHECK-NEXT:    [[BC1:%.*]] = trunc <4 x i64> %a to <4 x i32>
; CHECK-NEXT:    [[BC2:%.*]] = trunc <4 x i64> %b to <4 x i32>
; CHECK-NEXT:    [[SEL1:%.*]] = select <4 x i1> [[CMP]], <4 x i32> [[BC1]], <4 x i32> [[BC2]]
; CHECK-NEXT:    [[SEL2:%.*]] = select <4 x i1> [[CMP]], <4 x i32> [[BC2]], <4 x i32> [[BC1]]
; CHECK-NEXT:    store <4 x i32> [[SEL1]], <4 x i32>* %ptr1, align 16
; CHECK-NEXT:    store <4 x i32> [[SEL2]], <4 x i32>* %ptr2, align 16
; CHECK-NEXT:    ret void
;
  %cmp = fcmp olt <4 x float> %f1, %f2
  %bc1 = trunc <4 x i64> %a to <4 x i32>
  %bc2 = trunc <4 x i64> %b to <4 x i32>
  %sel1 = select <4 x i1> %cmp, <4 x i32> %bc1, <4 x i32> %bc2
  %sel2 = select <4 x i1> %cmp, <4 x i32> %bc2, <4 x i32> %bc1
  store <4 x i32> %sel1, <4 x i32>* %ptr1, align 16
  store <4 x i32> %sel2, <4 x i32>* %ptr2, align 16
  ret void
}

; PR8575

define i32 @test52(i32 %n, i32 %m) nounwind {
; CHECK-LABEL: @test52(
  %cmp = icmp sgt i32 %n, %m
  %. = select i1 %cmp, i32 1, i32 3
  %add = add nsw i32 %., 3
  %storemerge = select i1 %cmp, i32 %., i32 %add
; CHECK: select i1 %cmp, i32 1, i32 6
  ret i32 %storemerge
}

; PR9454
define i32 @test53(i32 %x) nounwind {
  %and = and i32 %x, 2
  %cmp = icmp eq i32 %and, %x
  %sel = select i1 %cmp, i32 2, i32 1
  ret i32 %sel
; CHECK-LABEL: @test53(
; CHECK: select i1 %cmp
; CHECK: ret
}

define i32 @test54(i32 %X, i32 %Y) {
  %A = ashr exact i32 %X, %Y
  %B = icmp eq i32 %A, 0
  %C = select i1 %B, i32 %A, i32 1
  ret i32 %C
; CHECK-LABEL: @test54(
; CHECK-NOT: ashr
; CHECK-NOT: select
; CHECK: icmp ne i32 %X, 0
; CHECK: zext
; CHECK: ret
}

define i1 @test55(i1 %X, i32 %Y, i32 %Z) {
  %A = ashr exact i32 %Y, %Z
  %B = select i1 %X, i32 %Y, i32 %A
  %C = icmp eq i32 %B, 0
  ret i1 %C
; CHECK-LABEL: @test55(
; CHECK-NOT: ashr
; CHECK-NOT: select
; CHECK: icmp eq
; CHECK: ret i1
}

define i32 @test56(i16 %x) nounwind {
  %tobool = icmp eq i16 %x, 0
  %conv = zext i16 %x to i32
  %cond = select i1 %tobool, i32 0, i32 %conv
  ret i32 %cond
; CHECK-LABEL: @test56(
; CHECK-NEXT: zext
; CHECK-NEXT: ret
}

define i32 @test57(i32 %x, i32 %y) nounwind {
  %and = and i32 %x, %y
  %tobool = icmp eq i32 %x, 0
  %.and = select i1 %tobool, i32 0, i32 %and
  ret i32 %.and
; CHECK-LABEL: @test57(
; CHECK-NEXT: and i32 %x, %y
; CHECK-NEXT: ret
}

define i32 @test58(i16 %x) nounwind {
  %tobool = icmp ne i16 %x, 1
  %conv = zext i16 %x to i32
  %cond = select i1 %tobool, i32 %conv, i32 1
  ret i32 %cond
; CHECK-LABEL: @test58(
; CHECK-NEXT: zext
; CHECK-NEXT: ret
}

define i32 @test59(i32 %x, i32 %y) nounwind {
  %and = and i32 %x, %y
  %tobool = icmp ne i32 %x, %y
  %.and = select i1 %tobool, i32 %and, i32 %y
  ret i32 %.and
; CHECK-LABEL: @test59(
; CHECK-NEXT: and i32 %x, %y
; CHECK-NEXT: ret
}

define i1 @test60(i32 %x, i1* %y) nounwind {
  %cmp = icmp eq i32 %x, 0
  %load = load i1, i1* %y, align 1
  %cmp1 = icmp slt i32 %x, 1
  %sel = select i1 %cmp, i1 %load, i1 %cmp1
  ret i1 %sel
; CHECK-LABEL: @test60(
; CHECK: select
}

@glbl = constant i32 10
define i32 @test61(i32* %ptr) {
  %A = load i32, i32* %ptr
  %B = icmp eq i32* %ptr, @glbl
  %C = select i1 %B, i32 %A, i32 10
  ret i32 %C
; CHECK-LABEL: @test61(
; CHECK: ret i32 10
}

; PR14131
define void @test64(i32 %p, i16 %b) noreturn nounwind {
entry:
  %p.addr.0.insert.mask = and i32 %p, -65536
  %conv2 = and i32 %p, 65535
  br i1 undef, label %lor.rhs, label %lor.end

lor.rhs:
  %p.addr.0.extract.trunc = trunc i32 %p.addr.0.insert.mask to i16
  %phitmp = zext i16 %p.addr.0.extract.trunc to i32
  br label %lor.end

lor.end:
  %t.1 = phi i32 [ 0, %entry ], [ %phitmp, %lor.rhs ]
  %conv6 = zext i16 %b to i32
  %div = udiv i32 %conv6, %t.1
  %tobool8 = icmp eq i32 %div, 0
  %cmp = icmp eq i32 %t.1, 0
  %cmp12 = icmp ult i32 %conv2, 2
  %cmp.sink = select i1 %tobool8, i1 %cmp12, i1 %cmp
  br i1 %cmp.sink, label %cond.end17, label %cond.false16

cond.false16:
  br label %cond.end17

cond.end17:
  br label %while.body

while.body:
  br label %while.body
; CHECK-LABEL: @test64(
; CHECK-NOT: select
}

@under_aligned = external global i32, align 1

define i32 @test76(i1 %flag, i32* %x) {
; The load here must not be speculated around the select. One side of the
; select is trivially dereferencable but may have a lower alignment than the
; load does.
; CHECK-LABEL: @test76(
; CHECK: store i32 0, i32* %x
; CHECK: %[[P:.*]] = select i1 %flag, i32* @under_aligned, i32* %x
; CHECK: load i32, i32* %[[P]]

  store i32 0, i32* %x
  %p = select i1 %flag, i32* @under_aligned, i32* %x
  %v = load i32, i32* %p
  ret i32 %v
}

declare void @scribble_on_i32(i32*)

define i32 @test77(i1 %flag, i32* %x) {
; The load here must not be speculated around the select. One side of the
; select is trivially dereferencable but may have a lower alignment than the
; load does.
; CHECK-LABEL: @test77(
; CHECK: %[[A:.*]] = alloca i32, align 1
; CHECK: call void @scribble_on_i32(i32* nonnull %[[A]])
; CHECK: store i32 0, i32* %x
; CHECK: %[[P:.*]] = select i1 %flag, i32* %[[A]], i32* %x
; CHECK: load i32, i32* %[[P]]

  %under_aligned = alloca i32, align 1
  call void @scribble_on_i32(i32* %under_aligned)
  store i32 0, i32* %x
  %p = select i1 %flag, i32* %under_aligned, i32* %x
  %v = load i32, i32* %p
  ret i32 %v
}

define i32 @test78(i1 %flag, i32* %x, i32* %y, i32* %z) {
; Test that we can speculate the loads around the select even when we can't
; fold the load completely away.
; CHECK-LABEL: @test78(
; CHECK:         %[[V1:.*]] = load i32, i32* %x
; CHECK-NEXT:    %[[V2:.*]] = load i32, i32* %y
; CHECK-NEXT:    %[[S:.*]] = select i1 %flag, i32 %[[V1]], i32 %[[V2]]
; CHECK-NEXT:    ret i32 %[[S]]
entry:
  store i32 0, i32* %x
  store i32 0, i32* %y
  ; Block forwarding by storing to %z which could alias either %x or %y.
  store i32 42, i32* %z
  %p = select i1 %flag, i32* %x, i32* %y
  %v = load i32, i32* %p
  ret i32 %v
}

define i32 @test78_deref(i1 %flag, i32* dereferenceable(4) %x, i32* dereferenceable(4) %y, i32* %z) {
; Test that we can speculate the loads around the select even when we can't
; fold the load completely away.
; CHECK-LABEL: @test78_deref(
; CHECK:         %[[V1:.*]] = load i32, i32* %x
; CHECK-NEXT:    %[[V2:.*]] = load i32, i32* %y
; CHECK-NEXT:    %[[S:.*]] = select i1 %flag, i32 %[[V1]], i32 %[[V2]]
; CHECK-NEXT:    ret i32 %[[S]]
entry:
  %p = select i1 %flag, i32* %x, i32* %y
  %v = load i32, i32* %p
  ret i32 %v
}

define i32 @test78_neg(i1 %flag, i32* %x, i32* %y, i32* %z) {
; The same as @test78 but we can't speculate the load because it can trap
; if under-aligned.
; CHECK-LABEL: @test78_neg(
; CHECK: %p = select i1 %flag, i32* %x, i32* %y
; CHECK-NEXT: %v = load i32, i32* %p, align 16
; CHECK-NEXT: ret i32 %v
entry:
  store i32 0, i32* %x
  store i32 0, i32* %y
  ; Block forwarding by storing to %z which could alias either %x or %y.
  store i32 42, i32* %z
  %p = select i1 %flag, i32* %x, i32* %y
  %v = load i32, i32* %p, align 16
  ret i32 %v
}

define i32 @test78_deref_neg(i1 %flag, i32* dereferenceable(2) %x, i32* dereferenceable(4) %y, i32* %z) {
; The same as @test78_deref but we can't speculate the load because
; one of the arguments is not sufficiently dereferenceable.
; CHECK-LABEL: @test78_deref_neg(
; CHECK: %p = select i1 %flag, i32* %x, i32* %y
; CHECK-NEXT: %v = load i32, i32* %p
; CHECK-NEXT: ret i32 %v
entry:
  %p = select i1 %flag, i32* %x, i32* %y
  %v = load i32, i32* %p
  ret i32 %v
}

define float @test79(i1 %flag, float* %x, i32* %y, i32* %z) {
; Test that we can speculate the loads around the select even when we can't
; fold the load completely away.
; CHECK-LABEL: @test79(
; CHECK:         %[[V1:.*]] = load float, float* %x
; CHECK-NEXT:    %[[V2:.*]] = load float, float* %y
; CHECK-NEXT:    %[[S:.*]] = select i1 %flag, float %[[V1]], float %[[V2]]
; CHECK-NEXT:    ret float %[[S]]
entry:
  %x1 = bitcast float* %x to i32*
  %y1 = bitcast i32* %y to float*
  store i32 0, i32* %x1
  store i32 0, i32* %y
  ; Block forwarding by storing to %z which could alias either %x or %y.
  store i32 42, i32* %z
  %p = select i1 %flag, float* %x, float* %y1
  %v = load float, float* %p
  ret float %v
}

define i32 @test80(i1 %flag) {
; Test that when we speculate the loads around the select they fold throug
; load->load folding and load->store folding.
; CHECK-LABEL: @test80(
; CHECK:         %[[X:.*]] = alloca i32
; CHECK-NEXT:    %[[Y:.*]] = alloca i32
; CHECK:         %[[V:.*]] = load i32, i32* %[[X]]
; CHECK-NEXT:    store i32 %[[V]], i32* %[[Y]]
; CHECK-NEXT:    ret i32 %[[V]]
entry:
  %x = alloca i32
  %y = alloca i32
  call void @scribble_on_i32(i32* %x)
  call void @scribble_on_i32(i32* %y)
  %tmp = load i32, i32* %x
  store i32 %tmp, i32* %y
  %p = select i1 %flag, i32* %x, i32* %y
  %v = load i32, i32* %p
  ret i32 %v
}

define float @test81(i1 %flag) {
; Test that we can speculate the load around the select even though they use
; differently typed pointers.
; CHECK-LABEL: @test81(
; CHECK:         %[[X:.*]] = alloca i32
; CHECK-NEXT:    %[[Y:.*]] = alloca i32
; CHECK:         %[[V:.*]] = load i32, i32* %[[X]]
; CHECK-NEXT:    store i32 %[[V]], i32* %[[Y]]
; CHECK-NEXT:    %[[C:.*]] = bitcast i32 %[[V]] to float
; CHECK-NEXT:    ret float %[[C]]
entry:
  %x = alloca float
  %y = alloca i32
  %x1 = bitcast float* %x to i32*
  %y1 = bitcast i32* %y to float*
  call void @scribble_on_i32(i32* %x1)
  call void @scribble_on_i32(i32* %y)
  %tmp = load i32, i32* %x1
  store i32 %tmp, i32* %y
  %p = select i1 %flag, float* %x, float* %y1
  %v = load float, float* %p
  ret float %v
}

define i32 @test82(i1 %flag) {
; Test that we can speculate the load around the select even though they use
; differently typed pointers.
; CHECK-LABEL: @test82(
; CHECK:         %[[X:.*]] = alloca float
; CHECK-NEXT:    %[[Y:.*]] = alloca i32
; CHECK-NEXT:    %[[X1:.*]] = bitcast float* %[[X]] to i32*
; CHECK-NEXT:    %[[Y1:.*]] = bitcast i32* %[[Y]] to float*
; CHECK:         %[[V:.*]] = load float, float* %[[X]]
; CHECK-NEXT:    store float %[[V]], float* %[[Y1]]
; CHECK-NEXT:    %[[C:.*]] = bitcast float %[[V]] to i32
; CHECK-NEXT:    ret i32 %[[C]]
entry:
  %x = alloca float
  %y = alloca i32
  %x1 = bitcast float* %x to i32*
  %y1 = bitcast i32* %y to float*
  call void @scribble_on_i32(i32* %x1)
  call void @scribble_on_i32(i32* %y)
  %tmp = load float, float* %x
  store float %tmp, float* %y1
  %p = select i1 %flag, i32* %x1, i32* %y
  %v = load i32, i32* %p
  ret i32 %v
}

declare void @scribble_on_i64(i64*)
declare void @scribble_on_i128(i128*)

define i8* @test83(i1 %flag) {
; Test that we can speculate the load around the select even though they use
; differently typed pointers and requires inttoptr casts.
; CHECK-LABEL: @test83(
; CHECK:         %[[X:.*]] = alloca i8*
; CHECK-NEXT:    %[[Y:.*]] = alloca i8*
; CHECK-DAG:     %[[X2:.*]] = bitcast i8** %[[X]] to i64*
; CHECK-DAG:     %[[Y2:.*]] = bitcast i8** %[[Y]] to i64*
; CHECK:         %[[V:.*]] = load i64, i64* %[[X2]]
; CHECK-NEXT:    store i64 %[[V]], i64* %[[Y2]]
; CHECK-NEXT:    %[[C:.*]] = inttoptr i64 %[[V]] to i8*
; CHECK-NEXT:    ret i8* %[[S]]
entry:
  %x = alloca i8*
  %y = alloca i64
  %x1 = bitcast i8** %x to i64*
  %y1 = bitcast i64* %y to i8**
  call void @scribble_on_i64(i64* %x1)
  call void @scribble_on_i64(i64* %y)
  %tmp = load i64, i64* %x1
  store i64 %tmp, i64* %y
  %p = select i1 %flag, i8** %x, i8** %y1
  %v = load i8*, i8** %p
  ret i8* %v
}

define i64 @test84(i1 %flag) {
; Test that we can speculate the load around the select even though they use
; differently typed pointers and requires a ptrtoint cast.
; CHECK-LABEL: @test84(
; CHECK:         %[[X:.*]] = alloca i8*
; CHECK-NEXT:    %[[Y:.*]] = alloca i8*
; CHECK:         %[[V:.*]] = load i8*, i8** %[[X]]
; CHECK-NEXT:    store i8* %[[V]], i8** %[[Y]]
; CHECK-NEXT:    %[[C:.*]] = ptrtoint i8* %[[V]] to i64
; CHECK-NEXT:    ret i64 %[[C]]
entry:
  %x = alloca i8*
  %y = alloca i64
  %x1 = bitcast i8** %x to i64*
  %y1 = bitcast i64* %y to i8**
  call void @scribble_on_i64(i64* %x1)
  call void @scribble_on_i64(i64* %y)
  %tmp = load i8*, i8** %x
  store i8* %tmp, i8** %y1
  %p = select i1 %flag, i64* %x1, i64* %y
  %v = load i64, i64* %p
  ret i64 %v
}

define i8* @test85(i1 %flag) {
; Test that we can't speculate the load around the select. The load of the
; pointer doesn't load all of the stored integer bits. We could fix this, but it
; would require endianness checks and other nastiness.
; CHECK-LABEL: @test85(
; CHECK:         %[[T:.*]] = load i128, i128*
; CHECK-NEXT:    store i128 %[[T]], i128*
; CHECK-NEXT:    %[[X:.*]] = load i8*, i8**
; CHECK-NEXT:    %[[Y:.*]] = load i8*, i8**
; CHECK-NEXT:    %[[V:.*]] = select i1 %flag, i8* %[[X]], i8* %[[Y]]
; CHECK-NEXT:    ret i8* %[[V]]
entry:
  %x = alloca [2 x i8*]
  %y = alloca i128
  %x1 = bitcast [2 x i8*]* %x to i8**
  %x2 = bitcast i8** %x1 to i128*
  %y1 = bitcast i128* %y to i8**
  call void @scribble_on_i128(i128* %x2)
  call void @scribble_on_i128(i128* %y)
  %tmp = load i128, i128* %x2
  store i128 %tmp, i128* %y
  %p = select i1 %flag, i8** %x1, i8** %y1
  %v = load i8*, i8** %p
  ret i8* %v
}

define i128 @test86(i1 %flag) {
; Test that we can't speculate the load around the select when the integer size
; is larger than the pointer size. The store of the pointer doesn't store to all
; the bits of the integer.
;
; CHECK-LABEL: @test86(
; CHECK:         %[[T:.*]] = load i8*, i8**
; CHECK-NEXT:    store i8* %[[T]], i8**
; CHECK-NEXT:    %[[X:.*]] = load i128, i128*
; CHECK-NEXT:    %[[Y:.*]] = load i128, i128*
; CHECK-NEXT:    %[[V:.*]] = select i1 %flag, i128 %[[X]], i128 %[[Y]]
; CHECK-NEXT:    ret i128 %[[V]]
entry:
  %x = alloca [2 x i8*]
  %y = alloca i128
  %x1 = bitcast [2 x i8*]* %x to i8**
  %x2 = bitcast i8** %x1 to i128*
  %y1 = bitcast i128* %y to i8**
  call void @scribble_on_i128(i128* %x2)
  call void @scribble_on_i128(i128* %y)
  %tmp = load i8*, i8** %x1
  store i8* %tmp, i8** %y1
  %p = select i1 %flag, i128* %x2, i128* %y
  %v = load i128, i128* %p
  ret i128 %v
}

define i32 @test_select_select0(i32 %a, i32 %r0, i32 %r1, i32 %v1, i32 %v2) {
  ; CHECK-LABEL: @test_select_select0(
  ; CHECK: %[[C0:.*]] = icmp sge i32 %a, %v1
  ; CHECK-NEXT: %[[C1:.*]] = icmp slt i32 %a, %v2
  ; CHECK-NEXT: %[[C:.*]] = and i1 %[[C1]], %[[C0]]
  ; CHECK-NEXT: %[[SEL:.*]] = select i1 %[[C]], i32 %r0, i32 %r1
  ; CHECK-NEXT: ret i32 %[[SEL]]
  %c0 = icmp sge i32 %a, %v1
  %s0 = select i1 %c0, i32 %r0, i32 %r1
  %c1 = icmp slt i32 %a, %v2
  %s1 = select i1 %c1, i32 %s0, i32 %r1
  ret i32 %s1
}

define i32 @test_select_select1(i32 %a, i32 %r0, i32 %r1, i32 %v1, i32 %v2) {
  ; CHECK-LABEL: @test_select_select1(
  ; CHECK: %[[C0:.*]] = icmp sge i32 %a, %v1
  ; CHECK-NEXT: %[[C1:.*]] = icmp slt i32 %a, %v2
  ; CHECK-NEXT: %[[C:.*]] = or i1 %[[C1]], %[[C0]]
  ; CHECK-NEXT: %[[SEL:.*]] = select i1 %[[C]], i32 %r0, i32 %r1
  ; CHECK-NEXT: ret i32 %[[SEL]]
  %c0 = icmp sge i32 %a, %v1
  %s0 = select i1 %c0, i32 %r0, i32 %r1
  %c1 = icmp slt i32 %a, %v2
  %s1 = select i1 %c1, i32 %r0, i32 %s0
  ret i32 %s1
}

define i32 @PR23757(i32 %x) {
; CHECK-LABEL: @PR23757
; CHECK:      %[[cmp:.*]] = icmp eq i32 %x, 2147483647
; CHECK-NEXT: %[[add:.*]] = add nsw i32 %x, 1
; CHECK-NEXT: %[[sel:.*]] = select i1 %[[cmp]], i32 -2147483648, i32 %[[add]]
; CHECK-NEXT: ret i32 %[[sel]]
  %cmp = icmp eq i32 %x, 2147483647
  %add = add nsw i32 %x, 1
  %sel = select i1 %cmp, i32 -2147483648, i32 %add
  ret i32 %sel
}

; max(max(~a, -1), -1) --> max(~a, -1)

define i32 @PR27137(i32 %a) {
; CHECK-LABEL: @PR27137(
; CHECK-NEXT:    [[NOT_A:%.*]] = xor i32 %a, -1
; CHECK-NEXT:    [[C0:%.*]] = icmp sgt i32 [[NOT_A]], -1
; CHECK-NEXT:    [[S0:%.*]] = select i1 [[C0]], i32 [[NOT_A]], i32 -1
; CHECK-NEXT:    ret i32 [[S0]]
;
  %not_a = xor i32 %a, -1
  %c0 = icmp slt i32 %a, 0
  %s0 = select i1 %c0, i32 %not_a, i32 -1
  %c1 = icmp sgt i32 %s0, -1
  %s1 = select i1 %c1, i32 %s0, i32 -1
  ret i32 %s1
}

define i32 @select_icmp_slt0_xor(i32 %x) {
; CHECK-LABEL: @select_icmp_slt0_xor(
; CHECK-NEXT:    [[TMP1:%.*]] = or i32 %x, -2147483648
; CHECK-NEXT:    ret i32 [[TMP1]]
;
  %cmp = icmp slt i32 %x, zeroinitializer
  %xor = xor i32 %x, 2147483648
  %x.xor = select i1 %cmp, i32 %x, i32 %xor
  ret i32 %x.xor
}

define <2 x i32> @select_icmp_slt0_xor_vec(<2 x i32> %x) {
; CHECK-LABEL: @select_icmp_slt0_xor_vec(
; CHECK-NEXT:    [[TMP1:%.*]] = or <2 x i32> %x, <i32 -2147483648, i32 -2147483648>
; CHECK-NEXT:    ret <2 x i32> [[TMP1]]
;
  %cmp = icmp slt <2 x i32> %x, zeroinitializer
  %xor = xor <2 x i32> %x, <i32 2147483648, i32 2147483648>
  %x.xor = select <2 x i1> %cmp, <2 x i32> %x, <2 x i32> %xor
  ret <2 x i32> %x.xor
}

define <4 x i32> @canonicalize_to_shuffle(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: @canonicalize_to_shuffle(
; CHECK-NEXT:    [[SEL:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 5, i32 6, i32 3>
; CHECK-NEXT:    ret <4 x i32> [[SEL]]
;
  %sel = select <4 x i1> <i1 true, i1 false, i1 false, i1 true>, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %sel
}

; Undef elements of the select condition may not be translated into undef elements of a shuffle mask
; because undef in a shuffle mask means we can return anything, not just one of the selected values.
; https://bugs.llvm.org/show_bug.cgi?id=32486

define <4 x i32> @undef_elts_in_condition(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: @undef_elts_in_condition(
; CHECK-NEXT:    [[SEL:%.*]] = select <4 x i1> <i1 true, i1 undef, i1 false, i1 undef>, <4 x i32> %a, <4 x i32> %b
; CHECK-NEXT:    ret <4 x i32> [[SEL]]
;
  %sel = select <4 x i1> <i1 true, i1 undef, i1 false, i1 undef>, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %sel
}

; Don't die or try if the condition mask is a constant expression or contains a constant expression.

@g = global i32 0

define <4 x i32> @cannot_canonicalize_to_shuffle1(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: @cannot_canonicalize_to_shuffle1(
; CHECK-NEXT:    [[SEL:%.*]] = select <4 x i1> bitcast (i4 ptrtoint (i32* @g to i4) to <4 x i1>), <4 x i32> %a, <4 x i32> %b
; CHECK-NEXT:    ret <4 x i32> [[SEL]]
;
  %sel = select <4 x i1> bitcast (i4 ptrtoint (i32* @g to i4) to <4 x i1>), <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %sel
}

define <4 x i32> @cannot_canonicalize_to_shuffle2(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: @cannot_canonicalize_to_shuffle2(
; CHECK-NEXT:    [[SEL:%.*]] = select <4 x i1> <i1 true, i1 undef, i1 false, i1 icmp sle (i16 ptrtoint (i32* @g to i16), i16 4)>, <4 x i32> %a, <4 x i32> %b
; CHECK-NEXT:    ret <4 x i32> [[SEL]]
;
  %sel = select <4 x i1> <i1 true, i1 undef, i1 false, i1 icmp sle (i16 ptrtoint (i32* @g to i16), i16 4)>, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %sel
}

declare void @llvm.assume(i1)

define i8 @assume_cond_true(i1 %cond, i8 %x, i8 %y) {
; CHECK-LABEL: @assume_cond_true(
; CHECK-NEXT:    call void @llvm.assume(i1 %cond)
; CHECK-NEXT:    ret i8 %x
;
  call void @llvm.assume(i1 %cond)
  %sel = select i1 %cond, i8 %x, i8 %y
  ret i8 %sel
}

; computeKnownBitsFromAssume() understands the 'not' of an assumed condition.

define i8 @assume_cond_false(i1 %cond, i8 %x, i8 %y) {
; CHECK-LABEL: @assume_cond_false(
; CHECK-NEXT:    [[NOTCOND:%.*]] = xor i1 %cond, true
; CHECK-NEXT:    call void @llvm.assume(i1 [[NOTCOND]])
; CHECK-NEXT:    ret i8 %y
;
  %notcond = xor i1 %cond, true
  call void @llvm.assume(i1 %notcond)
  %sel = select i1 %cond, i8 %x, i8 %y
  ret i8 %sel
}

