target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

; Optimize subtracts.
;
; RUN: opt < %s -instcombine -S | FileCheck %s

define i32 @test1(i32 %A) {
	%B = sub i32 %A, %A	
	ret i32 %B
; CHECK: @test1
; CHECK: ret i32 0
}

define i32 @test2(i32 %A) {
	%B = sub i32 %A, 0	
	ret i32 %B
; CHECK: @test2
; CHECK: ret i32 %A
}

define i32 @test3(i32 %A) {
	%B = sub i32 0, %A	
	%C = sub i32 0, %B	
	ret i32 %C
; CHECK: @test3
; CHECK: ret i32 %A
}

define i32 @test4(i32 %A, i32 %x) {
	%B = sub i32 0, %A	
	%C = sub i32 %x, %B	
	ret i32 %C
; CHECK: @test4
; CHECK: %C = add i32 %x, %A
; CHECK: ret i32 %C
}

define i32 @test5(i32 %A, i32 %B, i32 %C) {
	%D = sub i32 %B, %C	
	%E = sub i32 %A, %D	
	ret i32 %E
; CHECK: @test5
; CHECK: %D1 = sub i32 %C, %B
; CHECK: %E = add
; CHECK: ret i32 %E
}

define i32 @test6(i32 %A, i32 %B) {
	%C = and i32 %A, %B	
	%D = sub i32 %A, %C	
	ret i32 %D
; CHECK: @test6
; CHECK-NEXT: xor i32 %B, -1
; CHECK-NEXT: %D = and i32 
; CHECK-NEXT: ret i32 %D
}

define i32 @test7(i32 %A) {
	%B = sub i32 -1, %A	
	ret i32 %B
; CHECK: @test7
; CHECK: %B = xor i32 %A, -1
; CHECK: ret i32 %B
}

define i32 @test8(i32 %A) {
	%B = mul i32 9, %A	
	%C = sub i32 %B, %A	
	ret i32 %C
; CHECK: @test8
; CHECK: %C = shl i32 %A, 3
; CHECK: ret i32 %C
}

define i32 @test9(i32 %A) {
	%B = mul i32 3, %A	
	%C = sub i32 %A, %B	
	ret i32 %C
; CHECK: @test9
; CHECK: %C = mul i32 %A, -2
; CHECK: ret i32 %C
}

define i32 @test10(i32 %A, i32 %B) {
	%C = sub i32 0, %A	
	%D = sub i32 0, %B	
	%E = mul i32 %C, %D	
	ret i32 %E
; CHECK: @test10
; CHECK: %E = mul i32 %A, %B
; CHECK: ret i32 %E
}

define i32 @test10a(i32 %A) {
	%C = sub i32 0, %A	
	%E = mul i32 %C, 7	
	ret i32 %E
; CHECK: @test10a
; CHECK: %E = mul i32 %A, -7
; CHECK: ret i32 %E
}

define i1 @test11(i8 %A, i8 %B) {
	%C = sub i8 %A, %B	
	%cD = icmp ne i8 %C, 0	
	ret i1 %cD
; CHECK: @test11
; CHECK: %cD = icmp ne i8 %A, %B
; CHECK: ret i1 %cD
}

define i32 @test12(i32 %A) {
	%B = ashr i32 %A, 31	
	%C = sub i32 0, %B	
	ret i32 %C
; CHECK: @test12
; CHECK: %C = lshr i32 %A, 31
; CHECK: ret i32 %C
}

define i32 @test13(i32 %A) {
	%B = lshr i32 %A, 31	
	%C = sub i32 0, %B	
	ret i32 %C
; CHECK: @test13
; CHECK: %C = ashr i32 %A, 31
; CHECK: ret i32 %C
}

define i32 @test14(i32 %A) {
	%B = lshr i32 %A, 31	
	%C = bitcast i32 %B to i32	
	%D = sub i32 0, %C	
	ret i32 %D
; CHECK: @test14
; CHECK: %D = ashr i32 %A, 31
; CHECK: ret i32 %D
}

define i32 @test15(i32 %A, i32 %B) {
	%C = sub i32 0, %A	
	%D = srem i32 %B, %C	
	ret i32 %D
; CHECK: @test15
; CHECK: %D = srem i32 %B, %A 
; CHECK: ret i32 %D
}

define i32 @test16(i32 %A) {
	%X = sdiv i32 %A, 1123	
	%Y = sub i32 0, %X	
	ret i32 %Y
; CHECK: @test16
; CHECK: %Y = sdiv i32 %A, -1123
; CHECK: ret i32 %Y
}

; Can't fold subtract here because negation it might oveflow.
; PR3142
define i32 @test17(i32 %A) {
	%B = sub i32 0, %A	
	%C = sdiv i32 %B, 1234	
	ret i32 %C
; CHECK: @test17
; CHECK: %B = sub i32 0, %A
; CHECK: %C = sdiv i32 %B, 1234
; CHECK: ret i32 %C
}

define i64 @test18(i64 %Y) {
	%tmp.4 = shl i64 %Y, 2	
	%tmp.12 = shl i64 %Y, 2	
	%tmp.8 = sub i64 %tmp.4, %tmp.12	
	ret i64 %tmp.8
; CHECK: @test18
; CHECK: ret i64 0
}

define i32 @test19(i32 %X, i32 %Y) {
	%Z = sub i32 %X, %Y	
	%Q = add i32 %Z, %Y	
	ret i32 %Q
; CHECK: @test19
; CHECK: ret i32 %X
}

define i1 @test20(i32 %g, i32 %h) {
	%tmp.2 = sub i32 %g, %h	
	%tmp.4 = icmp ne i32 %tmp.2, %g	
	ret i1 %tmp.4
; CHECK: @test20
; CHECK: %tmp.4 = icmp ne i32 %h, 0
; CHECK: ret i1 %tmp.4
}

define i1 @test21(i32 %g, i32 %h) {
	%tmp.2 = sub i32 %g, %h	
	%tmp.4 = icmp ne i32 %tmp.2, %g		
        ret i1 %tmp.4
; CHECK: @test21
; CHECK: %tmp.4 = icmp ne i32 %h, 0
; CHECK: ret i1 %tmp.4
}

; PR2298
define zeroext i1 @test22(i32 %a, i32 %b)  nounwind  {
	%tmp2 = sub i32 0, %a	
	%tmp4 = sub i32 0, %b	
	%tmp5 = icmp eq i32 %tmp2, %tmp4	
	ret i1 %tmp5
; CHECK: @test22
; CHECK: %tmp5 = icmp eq i32 %b, %a
; CHECK: ret i1 %tmp5
}

; rdar://7362831
define i32 @test23(i8* %P, i64 %A){
  %B = getelementptr inbounds i8* %P, i64 %A
  %C = ptrtoint i8* %B to i64
  %D = trunc i64 %C to i32
  %E = ptrtoint i8* %P to i64
  %F = trunc i64 %E to i32
  %G = sub i32 %D, %F
  ret i32 %G
; CHECK: @test23
; CHECK-NEXT: = trunc i64 %A to i32
; CHECK-NEXT: ret i32
}

define i64 @test24(i8* %P, i64 %A){
  %B = getelementptr inbounds i8* %P, i64 %A
  %C = ptrtoint i8* %B to i64
  %E = ptrtoint i8* %P to i64
  %G = sub i64 %C, %E
  ret i64 %G
; CHECK: @test24
; CHECK-NEXT: ret i64 %A
}

define i64 @test24a(i8* %P, i64 %A){
  %B = getelementptr inbounds i8* %P, i64 %A
  %C = ptrtoint i8* %B to i64
  %E = ptrtoint i8* %P to i64
  %G = sub i64 %E, %C
  ret i64 %G
; CHECK: @test24a
; CHECK-NEXT: sub i64 0, %A
; CHECK-NEXT: ret i64 
}

@Arr = external global [42 x i16]

define i64 @test24b(i8* %P, i64 %A){
  %B = getelementptr inbounds [42 x i16]* @Arr, i64 0, i64 %A
  %C = ptrtoint i16* %B to i64
  %G = sub i64 %C, ptrtoint ([42 x i16]* @Arr to i64)
  ret i64 %G
; CHECK: @test24b
; CHECK-NEXT: shl nuw i64 %A, 1
; CHECK-NEXT: ret i64 
}


define i64 @test25(i8* %P, i64 %A){
  %B = getelementptr inbounds [42 x i16]* @Arr, i64 0, i64 %A
  %C = ptrtoint i16* %B to i64
  %G = sub i64 %C, ptrtoint (i16* getelementptr ([42 x i16]* @Arr, i64 1, i64 0) to i64)
  ret i64 %G
; CHECK: @test25
; CHECK-NEXT: shl nuw i64 %A, 1
; CHECK-NEXT: add i64 {{.*}}, -84
; CHECK-NEXT: ret i64 
}

define i32 @test26(i32 %x) {
  %shl = shl i32 3, %x
  %neg = sub i32 0, %shl
  ret i32 %neg
; CHECK: @test26
; CHECK-NEXT: shl i32 -3
; CHECK-NEXT: ret i32
}

define i32 @test27(i32 %x, i32 %y) {
  %mul = mul i32 %y, -8
  %sub = sub i32 %x, %mul
  ret i32 %sub
; CHECK: @test27
; CHECK-NEXT: shl i32 %y, 3
; CHECK-NEXT: add i32
; CHECK-NEXT: ret i32
}

define i32 @test28(i32 %x, i32 %y, i32 %z) {
  %neg = sub i32 0, %z
  %mul = mul i32 %neg, %y
  %sub = sub i32 %x, %mul
  ret i32 %sub
; CHECK: @test28
; CHECK-NEXT: mul i32 %z, %y
; CHECK-NEXT: add i32
; CHECK-NEXT: ret i32
}

define i64 @test29(i8* %foo, i64 %i, i64 %j) {
  %gep1 = getelementptr inbounds i8* %foo, i64 %i
  %gep2 = getelementptr inbounds i8* %foo, i64 %j
  %cast1 = ptrtoint i8* %gep1 to i64
  %cast2 = ptrtoint i8* %gep2 to i64
  %sub = sub i64 %cast1, %cast2
  ret i64 %sub
; CHECK: @test29
; CHECK-NEXT: sub i64 %i, %j
; CHECK-NEXT: ret i64
}

define i64 @test30(i8* %foo, i64 %i, i64 %j) {
  %bit = bitcast i8* %foo to i32*
  %gep1 = getelementptr inbounds i32* %bit, i64 %i
  %gep2 = getelementptr inbounds i8* %foo, i64 %j
  %cast1 = ptrtoint i32* %gep1 to i64
  %cast2 = ptrtoint i8* %gep2 to i64
  %sub = sub i64 %cast1, %cast2
  ret i64 %sub
; CHECK: @test30
; CHECK-NEXT: %gep1.idx = shl nuw i64 %i, 2
; CHECK-NEXT: sub i64 %gep1.idx, %j
; CHECK-NEXT: ret i64
}
