; This test makes sure that these instructions are properly eliminated.
; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

define i32 @test1(i32 %A) {
        %B = or i32 %A, 0
        ret i32 %B
; CHECK-LABEL: @test1(
; CHECK: ret i32 %A
}

define i32 @test2(i32 %A) {
        %B = or i32 %A, -1 
        ret i32 %B
; CHECK-LABEL: @test2(
; CHECK: ret i32 -1
}

define i8 @test2a(i8 %A) {
        %B = or i8 %A, -1  
        ret i8 %B
; CHECK-LABEL: @test2a(
; CHECK: ret i8 -1
}

define i1 @test3(i1 %A) {
        %B = or i1 %A, false
        ret i1 %B
; CHECK-LABEL: @test3(
; CHECK: ret i1 %A
}

define i1 @test4(i1 %A) {
        %B = or i1 %A, true 
        ret i1 %B
; CHECK-LABEL: @test4(
; CHECK: ret i1 true
}

define i1 @test5(i1 %A) {
        %B = or i1 %A, %A   
        ret i1 %B
; CHECK-LABEL: @test5(
; CHECK: ret i1 %A
}

define i32 @test6(i32 %A) {
        %B = or i32 %A, %A  
        ret i32 %B
; CHECK-LABEL: @test6(
; CHECK: ret i32 %A
}

; A | ~A == -1
define i32 @test7(i32 %A) {
        %NotA = xor i32 -1, %A
        %B = or i32 %A, %NotA
        ret i32 %B
; CHECK-LABEL: @test7(
; CHECK: ret i32 -1
}

define i8 @test8(i8 %A) {
        %B = or i8 %A, -2
        %C = or i8 %B, 1
        ret i8 %C
; CHECK-LABEL: @test8(
; CHECK: ret i8 -1
}

; Test that (A|c1)|(B|c2) == (A|B)|(c1|c2)
define i8 @test9(i8 %A, i8 %B) {
        %C = or i8 %A, 1
        %D = or i8 %B, -2
        %E = or i8 %C, %D
        ret i8 %E
; CHECK-LABEL: @test9(
; CHECK: ret i8 -1
}

define i8 @test10(i8 %A) {
        %B = or i8 %A, 1
        %C = and i8 %B, -2
        ; (X & C1) | C2 --> (X | C2) & (C1|C2)
        %D = or i8 %C, -2
        ret i8 %D
; CHECK-LABEL: @test10(
; CHECK: ret i8 -2
}

define i8 @test11(i8 %A) {
        %B = or i8 %A, -2
        %C = xor i8 %B, 13
        ; (X ^ C1) | C2 --> (X | C2) ^ (C1&~C2)
        %D = or i8 %C, 1
        %E = xor i8 %D, 12
        ret i8 %E
; CHECK-LABEL: @test11(
; CHECK: ret i8 -1
}

define i32 @test12(i32 %A) {
        ; Should be eliminated
        %B = or i32 %A, 4
        %C = and i32 %B, 8
        ret i32 %C
; CHECK-LABEL: @test12(
; CHECK: %C = and i32 %A, 8
; CHECK: ret i32 %C
}

define i32 @test13(i32 %A) {
        %B = or i32 %A, 12
        ; Always equal to 8
        %C = and i32 %B, 8
        ret i32 %C
; CHECK-LABEL: @test13(
; CHECK: ret i32 8
}

define i1 @test14(i32 %A, i32 %B) {
        %C1 = icmp ult i32 %A, %B
        %C2 = icmp ugt i32 %A, %B
        ; (A < B) | (A > B) === A != B
        %D = or i1 %C1, %C2
        ret i1 %D
; CHECK-LABEL: @test14(
; CHECK: icmp ne i32 %A, %B
; CHECK: ret i1
}

define i1 @test15(i32 %A, i32 %B) {
        %C1 = icmp ult i32 %A, %B
        %C2 = icmp eq i32 %A, %B
        ; (A < B) | (A == B) === A <= B
        %D = or i1 %C1, %C2
        ret i1 %D
; CHECK-LABEL: @test15(
; CHECK:  icmp ule i32 %A, %B
; CHECK: ret i1
}

define i32 @test16(i32 %A) {
        %B = and i32 %A, 1
        ; -2 = ~1
        %C = and i32 %A, -2
        ; %D = and int %B, -1 == %B
        %D = or i32 %B, %C
        ret i32 %D
; CHECK-LABEL: @test16(
; CHECK: ret i32 %A
}

define i32 @test17(i32 %A) {
        %B = and i32 %A, 1
        %C = and i32 %A, 4
        ; %D = and int %B, 5
        %D = or i32 %B, %C
        ret i32 %D
; CHECK-LABEL: @test17(
; CHECK: %D = and i32 %A, 5
; CHECK: ret i32 %D
}

define i1 @test18(i32 %A) {
        %B = icmp sge i32 %A, 100
        %C = icmp slt i32 %A, 50
        ;; (A-50) >u 50
        %D = or i1 %B, %C
        ret i1 %D
; CHECK-LABEL: @test18(
; CHECK: add i32
; CHECK:  icmp ugt 
; CHECK: ret i1 
}

define i1 @test19(i32 %A) {
        %B = icmp eq i32 %A, 50
        %C = icmp eq i32 %A, 51
        ;; (A&-2) == 50
        %D = or i1 %B, %C
        ret i1 %D
; CHECK-LABEL: @test19(
; CHECK: or i32
; CHECK: icmp eq 
; CHECK: ret i1
}

define i32 @test20(i32 %x) {
        %y = and i32 %x, 123
        %z = or i32 %y, %x
        ret i32 %z
; CHECK-LABEL: @test20(
; CHECK: ret i32 %x
}

define i32 @test21(i32 %tmp.1) {
        %tmp.1.mask1 = add i32 %tmp.1, 2
        %tmp.3 = and i32 %tmp.1.mask1, -2
        %tmp.5 = and i32 %tmp.1, 1
        ;; add tmp.1, 2
        %tmp.6 = or i32 %tmp.5, %tmp.3
        ret i32 %tmp.6
; CHECK-LABEL: @test21(
; CHECK:   add i32 %{{[^,]*}}, 2
; CHECK:   ret i32 
}

define i32 @test22(i32 %B) {
        %ELIM41 = and i32 %B, 1
        %ELIM7 = and i32 %B, -2
        %ELIM5 = or i32 %ELIM41, %ELIM7
        ret i32 %ELIM5
; CHECK-LABEL: @test22(
; CHECK: ret i32 %B
}

define i16 @test23(i16 %A) {
        %B = lshr i16 %A, 1
        ;; fold or into xor
        %C = or i16 %B, -32768
        %D = xor i16 %C, 8193
        ret i16 %D
; CHECK-LABEL: @test23(
; CHECK:   %B = lshr i16 %A, 1
; CHECK:   %D = xor i16 %B, -24575
; CHECK:   ret i16 %D
}

; PR1738
define i1 @test24(double %X, double %Y) {
        %tmp9 = fcmp uno double %X, 0.000000e+00                ; <i1> [#uses=1]
        %tmp13 = fcmp uno double %Y, 0.000000e+00               ; <i1> [#uses=1]
        %bothcond = or i1 %tmp13, %tmp9         ; <i1> [#uses=1]
        ret i1 %bothcond
        
; CHECK-LABEL: @test24(
; CHECK:    = fcmp uno double %Y, %X
; CHECK:   ret i1 
}

; PR3266 & PR5276
define i1 @test25(i32 %A, i32 %B) {
  %C = icmp eq i32 %A, 0
  %D = icmp eq i32 %B, 57
  %E = or i1 %C, %D
  %F = xor i1 %E, -1
  ret i1 %F

; CHECK-LABEL: @test25(
; CHECK: icmp ne i32 %A, 0
; CHECK-NEXT: icmp ne i32 %B, 57
; CHECK-NEXT:  %F = and i1 
; CHECK-NEXT:  ret i1 %F
}

; PR5634
define i1 @test26(i32 %A, i32 %B) {
        %C1 = icmp eq i32 %A, 0
        %C2 = icmp eq i32 %B, 0
        ; (A == 0) & (A == 0)   -->   (A|B) == 0
        %D = and i1 %C1, %C2
        ret i1 %D
; CHECK-LABEL: @test26(
; CHECK: or i32 %A, %B
; CHECK: icmp eq i32 {{.*}}, 0
; CHECK: ret i1 
}

define i1 @test27(i32* %A, i32* %B) {
  %C1 = ptrtoint i32* %A to i32
  %C2 = ptrtoint i32* %B to i32
  %D = or i32 %C1, %C2
  %E = icmp eq i32 %D, 0
  ret i1 %E
; CHECK-LABEL: @test27(
; CHECK: icmp eq i32* %A, null
; CHECK: icmp eq i32* %B, null
; CHECK: and i1
; CHECK: ret i1
}

; PR5634
define i1 @test28(i32 %A, i32 %B) {
        %C1 = icmp ne i32 %A, 0
        %C2 = icmp ne i32 %B, 0
        ; (A != 0) | (A != 0)   -->   (A|B) != 0
        %D = or i1 %C1, %C2
        ret i1 %D
; CHECK-LABEL: @test28(
; CHECK: or i32 %A, %B
; CHECK: icmp ne i32 {{.*}}, 0
; CHECK: ret i1 
}

define i1 @test29(i32* %A, i32* %B) {
  %C1 = ptrtoint i32* %A to i32
  %C2 = ptrtoint i32* %B to i32
  %D = or i32 %C1, %C2
  %E = icmp ne i32 %D, 0
  ret i1 %E
; CHECK-LABEL: @test29(
; CHECK: icmp ne i32* %A, null
; CHECK: icmp ne i32* %B, null
; CHECK: or i1
; CHECK: ret i1
}

; PR4216
define i32 @test30(i32 %A) {
entry:
  %B = or i32 %A, 32962
  %C = and i32 %A, -65536
  %D = and i32 %B, 40186
  %E = or i32 %D, %C
  ret i32 %E
; CHECK-LABEL: @test30(
; CHECK: %D = and i32 %A, -58312
; CHECK: %E = or i32 %D, 32962
; CHECK: ret i32 %E
}

; PR4216
define i64 @test31(i64 %A) nounwind readnone ssp noredzone {
  %B = or i64 %A, 194
  %D = and i64 %B, 250

  %C = or i64 %A, 32768
  %E = and i64 %C, 4294941696

  %F = or i64 %D, %E
  ret i64 %F
; CHECK-LABEL: @test31(
; CHECK-NEXT: %E = and i64 %A, 4294908984
; CHECK-NEXT: %F = or i64 %E, 32962
; CHECK-NEXT: ret i64 %F
}

define <4 x i32> @test32(<4 x i1> %and.i1352, <4 x i32> %vecinit6.i176, <4 x i32> %vecinit6.i191) {
  %and.i135 = sext <4 x i1> %and.i1352 to <4 x i32> ; <<4 x i32>> [#uses=2]
  %and.i129 = and <4 x i32> %vecinit6.i176, %and.i135 ; <<4 x i32>> [#uses=1]
  %neg.i = xor <4 x i32> %and.i135, <i32 -1, i32 -1, i32 -1, i32 -1> ; <<4 x i32>> [#uses=1]
  %and.i = and <4 x i32> %vecinit6.i191, %neg.i   ; <<4 x i32>> [#uses=1]
  %or.i = or <4 x i32> %and.i, %and.i129          ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %or.i
; codegen is mature enough to handle vector selects.
; CHECK-LABEL: @test32(
; CHECK: select <4 x i1> %and.i1352, <4 x i32> %vecinit6.i176, <4 x i32> %vecinit6.i191
}

define i1 @test33(i1 %X, i1 %Y) {
  %a = or i1 %X, %Y
  %b = or i1 %a, %X
  ret i1 %b
; CHECK-LABEL: @test33(
; CHECK-NEXT: or i1 %X, %Y
; CHECK-NEXT: ret
}

define i32 @test34(i32 %X, i32 %Y) {
  %a = or i32 %X, %Y
  %b = or i32 %Y, %a
  ret i32 %b
; CHECK-LABEL: @test34(
; CHECK-NEXT: or i32 %X, %Y
; CHECK-NEXT: ret
}

define i32 @test35(i32 %a, i32 %b) {
  %1 = or i32 %a, 1135
  %2 = or i32 %1, %b
  ret i32 %2
  ; CHECK-LABEL: @test35(
  ; CHECK-NEXT: or i32 %a, %b
  ; CHECK-NEXT: or i32 %1, 1135
}

define i1 @test36(i32 %x) {
  %cmp1 = icmp eq i32 %x, 23
  %cmp2 = icmp eq i32 %x, 24
  %ret1 = or i1 %cmp1, %cmp2
  %cmp3 = icmp eq i32 %x, 25
  %ret2 = or i1 %ret1, %cmp3
  ret i1 %ret2
; CHECK-LABEL: @test36(
; CHECK-NEXT: %x.off = add i32 %x, -23
; CHECK-NEXT: icmp ult i32 %x.off, 3
; CHECK-NEXT: ret i1
}

define i32 @test37(i32* %xp, i32 %y) {
; CHECK-LABEL: @test37(
; CHECK: select i1 %tobool, i32 -1, i32 %x
  %tobool = icmp ne i32 %y, 0
  %sext = sext i1 %tobool to i32
  %x = load i32, i32* %xp
  %or = or i32 %sext, %x
  ret i32 %or
}

define i32 @test38(i32* %xp, i32 %y) {
; CHECK-LABEL: @test38(
; CHECK: select i1 %tobool, i32 -1, i32 %x
  %tobool = icmp ne i32 %y, 0
  %sext = sext i1 %tobool to i32
  %x = load i32, i32* %xp
  %or = or i32 %x, %sext
  ret i32 %or
}

define i32 @test39(i32 %a, i32 %b) {
; CHECK-LABEL: test39(
; CHECK-NEXT: %or = or i32 %a, %b
 %xor = xor i32 %a, -1
 %and = and i32 %xor, %b
 %or = or i32 %and, %a
 ret i32 %or
}

define i32 @test40(i32 %a, i32 %b) {
; CHECK-LABEL: test40(
; CHECK-NEXT:   %1 = xor i32 %a, -1 
; CHECK-NEXT: %or = or i32 %1, %b
 %and = and i32 %a, %b
 %xor = xor i32 %a, -1
 %or = or i32 %and, %xor
 ret i32 %or
}

define i32 @test41(i32 %a, i32 %b) {
; CHECK-LABEL: test41(
; CHECK-NEXT: %1 = xor i32 %a, -1
; CHECK-NEXT: %or = xor i32 %1, %b
 %and = and i32 %a, %b
 %nega = xor i32 %a, -1
 %xor = xor i32 %nega, %b
 %or = or i32 %and, %xor
 ret i32 %or
}

define i32 @test42(i32 %a, i32 %b) {
; CHECK-LABEL: test42(
; CHECK-NEXT: %1 = xor i32 %a, -1
; CHECK-NEXT: %or = xor i32 %1, %b
 %nega = xor i32 %a, -1
 %xor = xor i32 %nega, %b
 %and = and i32 %a, %b
 %or = or i32 %xor, %and
 ret i32 %or
}

define i32 @test43(i32 %a, i32 %b) {
; CHECK-LABEL: test43(
; CHECK-NEXT: %or = xor i32 %a, %b
 %neg = xor i32 %b, -1
 %and = and i32 %a, %neg
 %xor = xor i32 %a, %b
 %or = or i32 %and, %xor
 ret i32 %or
}

define i32 @test44(i32 %a, i32 %b) {
; CHECK-LABEL: test44(
; CHECK-NEXT: %or = xor i32 %a, %b
 %xor = xor i32 %a, %b
 %neg = xor i32 %b, -1
 %and = and i32 %a, %neg
 %or = or i32 %xor, %and
 ret i32 %or
}

define i32 @test45(i32 %x, i32 %y, i32 %z) {
; CHECK-LABEL: test45(
; CHECK-NEXT: %1 = and i32 %x, %z
; CHECK-NEXT: %or1 = or i32 %1, %y
; CHECK-NEXT: ret i32 %or1
  %or = or i32 %y, %z
  %and = and i32 %x, %or
  %or1 = or i32 %and, %y
  ret i32 %or1
}

define i1 @test46(i8 signext %c)  {
  %c.off = add i8 %c, -97
  %cmp1 = icmp ult i8 %c.off, 26
  %c.off17 = add i8 %c, -65
  %cmp2 = icmp ult i8 %c.off17, 26
  %or = or i1 %cmp1, %cmp2
  ret i1 %or
; CHECK-LABEL: @test46(
; CHECK-NEXT:  and i8 %c, -33
; CHECK-NEXT:  add i8 %1, -65
; CHECK-NEXT:  icmp ult i8 %2, 26
}

define i1 @test47(i8 signext %c)  {
  %c.off = add i8 %c, -65
  %cmp1 = icmp ule i8 %c.off, 26
  %c.off17 = add i8 %c, -97
  %cmp2 = icmp ule i8 %c.off17, 26
  %or = or i1 %cmp1, %cmp2
  ret i1 %or
; CHECK-LABEL: @test47(
; CHECK-NEXT:  and i8 %c, -33
; CHECK-NEXT:  add i8 %1, -65
; CHECK-NEXT:  icmp ult i8 %2, 27
}

define i1 @test48(i64 %x, i1 %b) {
  %1 = icmp ult i64 %x, 2305843009213693952
  %2 = icmp ugt i64 %x, 2305843009213693951
  %.b = or i1 %2, %b
  %3 = or i1 %1, %.b
  ret i1 %3
; CHECK-LABEL: @test48(
; CHECK-NEXT:  ret i1 true
}
