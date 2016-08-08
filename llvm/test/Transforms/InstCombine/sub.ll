target datalayout = "e-p:64:64:64-p1:16:16:16-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

; Optimize subtracts.
;
; RUN: opt < %s -instcombine -S | FileCheck %s

define i32 @test1(i32 %A) {
	%B = sub i32 %A, %A
	ret i32 %B
; CHECK-LABEL: @test1(
; CHECK: ret i32 0
}

define i32 @test2(i32 %A) {
	%B = sub i32 %A, 0
	ret i32 %B
; CHECK-LABEL: @test2(
; CHECK: ret i32 %A
}

define i32 @test3(i32 %A) {
	%B = sub i32 0, %A
	%C = sub i32 0, %B
	ret i32 %C
; CHECK-LABEL: @test3(
; CHECK: ret i32 %A
}

define i32 @test4(i32 %A, i32 %x) {
	%B = sub i32 0, %A
	%C = sub i32 %x, %B
	ret i32 %C
; CHECK-LABEL: @test4(
; CHECK: %C = add i32 %x, %A
; CHECK: ret i32 %C
}

define i32 @test5(i32 %A, i32 %B, i32 %C) {
	%D = sub i32 %B, %C
	%E = sub i32 %A, %D
	ret i32 %E
; CHECK-LABEL: @test5(
; CHECK: %D1 = sub i32 %C, %B
; CHECK: %E = add
; CHECK: ret i32 %E
}

define i32 @test6(i32 %A, i32 %B) {
	%C = and i32 %A, %B
	%D = sub i32 %A, %C
	ret i32 %D
; CHECK-LABEL: @test6(
; CHECK-NEXT: xor i32 %B, -1
; CHECK-NEXT: %D = and i32
; CHECK-NEXT: ret i32 %D
}

define i32 @test7(i32 %A) {
	%B = sub i32 -1, %A
	ret i32 %B
; CHECK-LABEL: @test7(
; CHECK: %B = xor i32 %A, -1
; CHECK: ret i32 %B
}

define i32 @test8(i32 %A) {
	%B = mul i32 9, %A
	%C = sub i32 %B, %A
	ret i32 %C
; CHECK-LABEL: @test8(
; CHECK: %C = shl i32 %A, 3
; CHECK: ret i32 %C
}

define i32 @test9(i32 %A) {
	%B = mul i32 3, %A
	%C = sub i32 %A, %B
	ret i32 %C
; CHECK-LABEL: @test9(
; CHECK: %C = mul i32 %A, -2
; CHECK: ret i32 %C
}

define i32 @test10(i32 %A, i32 %B) {
	%C = sub i32 0, %A
	%D = sub i32 0, %B
	%E = mul i32 %C, %D
	ret i32 %E
; CHECK-LABEL: @test10(
; CHECK: %E = mul i32 %A, %B
; CHECK: ret i32 %E
}

define i32 @test10a(i32 %A) {
	%C = sub i32 0, %A
	%E = mul i32 %C, 7
	ret i32 %E
; CHECK-LABEL: @test10a(
; CHECK: %E = mul i32 %A, -7
; CHECK: ret i32 %E
}

define i1 @test11(i8 %A, i8 %B) {
; CHECK-LABEL: @test11(
; CHECK-NEXT:    [[D:%.*]] = icmp ne i8 %A, %B
; CHECK-NEXT:    ret i1 [[D]]
;
  %C = sub i8 %A, %B
  %D = icmp ne i8 %C, 0
  ret i1 %D
}

define <2 x i1> @test11vec(<2 x i8> %A, <2 x i8> %B) {
; CHECK-LABEL: @test11vec(
; CHECK-NEXT:    [[D:%.*]] = icmp ne <2 x i8> %A, %B
; CHECK-NEXT:    ret <2 x i1> [[D]]
;
  %C = sub <2 x i8> %A, %B
  %D = icmp ne <2 x i8> %C, zeroinitializer
  ret <2 x i1> %D
}

define i32 @test12(i32 %A) {
	%B = ashr i32 %A, 31
	%C = sub i32 0, %B
	ret i32 %C
; CHECK-LABEL: @test12(
; CHECK: %C = lshr i32 %A, 31
; CHECK: ret i32 %C
}

define i32 @test13(i32 %A) {
	%B = lshr i32 %A, 31
	%C = sub i32 0, %B
	ret i32 %C
; CHECK-LABEL: @test13(
; CHECK: %C = ashr i32 %A, 31
; CHECK: ret i32 %C
}

define i32 @test14(i32 %A) {
	%B = lshr i32 %A, 31
	%C = bitcast i32 %B to i32
	%D = sub i32 0, %C
	ret i32 %D
; CHECK-LABEL: @test14(
; CHECK: %D = ashr i32 %A, 31
; CHECK: ret i32 %D
}

define i32 @test15(i32 %A, i32 %B) {
	%C = sub i32 0, %A
	%D = srem i32 %B, %C
	ret i32 %D
; CHECK-LABEL: @test15(
; CHECK:      %[[sub:.*]] = sub i32 0, %A
; CHECK-NEXT: %[[rem:.*]] = srem i32 %B, %[[sub]]
; CHECK: ret i32 %[[rem]]
}

define i32 @test16(i32 %A) {
	%X = sdiv i32 %A, 1123
	%Y = sub i32 0, %X
	ret i32 %Y
; CHECK-LABEL: @test16(
; CHECK: %Y = sdiv i32 %A, -1123
; CHECK: ret i32 %Y
}

; Can't fold subtract here because negation it might oveflow.
; PR3142
define i32 @test17(i32 %A) {
	%B = sub i32 0, %A
	%C = sdiv i32 %B, 1234
	ret i32 %C
; CHECK-LABEL: @test17(
; CHECK: %B = sub i32 0, %A
; CHECK: %C = sdiv i32 %B, 1234
; CHECK: ret i32 %C
}

define i64 @test18(i64 %Y) {
	%tmp.4 = shl i64 %Y, 2
	%tmp.12 = shl i64 %Y, 2
	%tmp.8 = sub i64 %tmp.4, %tmp.12
	ret i64 %tmp.8
; CHECK-LABEL: @test18(
; CHECK: ret i64 0
}

define i32 @test19(i32 %X, i32 %Y) {
	%Z = sub i32 %X, %Y
	%Q = add i32 %Z, %Y
	ret i32 %Q
; CHECK-LABEL: @test19(
; CHECK: ret i32 %X
}

define i1 @test20(i32 %g, i32 %h) {
	%tmp.2 = sub i32 %g, %h
	%tmp.4 = icmp ne i32 %tmp.2, %g
	ret i1 %tmp.4
; CHECK-LABEL: @test20(
; CHECK: %tmp.4 = icmp ne i32 %h, 0
; CHECK: ret i1 %tmp.4
}

define i1 @test21(i32 %g, i32 %h) {
	%tmp.2 = sub i32 %g, %h
	%tmp.4 = icmp ne i32 %tmp.2, %g
        ret i1 %tmp.4
; CHECK-LABEL: @test21(
; CHECK: %tmp.4 = icmp ne i32 %h, 0
; CHECK: ret i1 %tmp.4
}

; PR2298
define zeroext i1 @test22(i32 %a, i32 %b)  nounwind  {
	%tmp2 = sub i32 0, %a
	%tmp4 = sub i32 0, %b
	%tmp5 = icmp eq i32 %tmp2, %tmp4
	ret i1 %tmp5
; CHECK-LABEL: @test22(
; CHECK: %tmp5 = icmp eq i32 %b, %a
; CHECK: ret i1 %tmp5
}

; rdar://7362831
define i32 @test23(i8* %P, i64 %A){
  %B = getelementptr inbounds i8, i8* %P, i64 %A
  %C = ptrtoint i8* %B to i64
  %D = trunc i64 %C to i32
  %E = ptrtoint i8* %P to i64
  %F = trunc i64 %E to i32
  %G = sub i32 %D, %F
  ret i32 %G
; CHECK-LABEL: @test23(
; CHECK-NEXT: = trunc i64 %A to i32
; CHECK-NEXT: ret i32
}

define i8 @test23_as1(i8 addrspace(1)* %P, i16 %A) {
; CHECK: @test23_as1
; CHECK-NEXT: = trunc i16 %A to i8
; CHECK-NEXT: ret i8
  %B = getelementptr inbounds i8, i8 addrspace(1)* %P, i16 %A
  %C = ptrtoint i8 addrspace(1)* %B to i16
  %D = trunc i16 %C to i8
  %E = ptrtoint i8 addrspace(1)* %P to i16
  %F = trunc i16 %E to i8
  %G = sub i8 %D, %F
  ret i8 %G
}

define i64 @test24(i8* %P, i64 %A){
  %B = getelementptr inbounds i8, i8* %P, i64 %A
  %C = ptrtoint i8* %B to i64
  %E = ptrtoint i8* %P to i64
  %G = sub i64 %C, %E
  ret i64 %G
; CHECK-LABEL: @test24(
; CHECK-NEXT: ret i64 %A
}

define i16 @test24_as1(i8 addrspace(1)* %P, i16 %A) {
; CHECK: @test24_as1
; CHECK-NEXT: ret i16 %A
  %B = getelementptr inbounds i8, i8 addrspace(1)* %P, i16 %A
  %C = ptrtoint i8 addrspace(1)* %B to i16
  %E = ptrtoint i8 addrspace(1)* %P to i16
  %G = sub i16 %C, %E
  ret i16 %G
}

define i64 @test24a(i8* %P, i64 %A){
  %B = getelementptr inbounds i8, i8* %P, i64 %A
  %C = ptrtoint i8* %B to i64
  %E = ptrtoint i8* %P to i64
  %G = sub i64 %E, %C
  ret i64 %G
; CHECK-LABEL: @test24a(
; CHECK-NEXT: sub i64 0, %A
; CHECK-NEXT: ret i64
}

define i16 @test24a_as1(i8 addrspace(1)* %P, i16 %A) {
; CHECK: @test24a_as1
; CHECK-NEXT: sub i16 0, %A
; CHECK-NEXT: ret i16
  %B = getelementptr inbounds i8, i8 addrspace(1)* %P, i16 %A
  %C = ptrtoint i8 addrspace(1)* %B to i16
  %E = ptrtoint i8 addrspace(1)* %P to i16
  %G = sub i16 %E, %C
  ret i16 %G
}


@Arr = external global [42 x i16]

define i64 @test24b(i8* %P, i64 %A){
  %B = getelementptr inbounds [42 x i16], [42 x i16]* @Arr, i64 0, i64 %A
  %C = ptrtoint i16* %B to i64
  %G = sub i64 %C, ptrtoint ([42 x i16]* @Arr to i64)
  ret i64 %G
; CHECK-LABEL: @test24b(
; CHECK-NEXT: shl nuw i64 %A, 1
; CHECK-NEXT: ret i64
}


define i64 @test25(i8* %P, i64 %A){
  %B = getelementptr inbounds [42 x i16], [42 x i16]* @Arr, i64 0, i64 %A
  %C = ptrtoint i16* %B to i64
  %G = sub i64 %C, ptrtoint (i16* getelementptr ([42 x i16], [42 x i16]* @Arr, i64 1, i64 0) to i64)
  ret i64 %G
; CHECK-LABEL: @test25(
; CHECK-NEXT: shl nuw i64 %A, 1
; CHECK-NEXT: add i64 {{.*}}, -84
; CHECK-NEXT: ret i64
}

@Arr_as1 = external addrspace(1) global [42 x i16]

define i16 @test25_as1(i8 addrspace(1)* %P, i64 %A) {
; CHECK: @test25_as1
; CHECK-NEXT: %1 = trunc i64 %A to i16
; CHECK-NEXT: shl nuw i16 %1, 1
; CHECK-NEXT: add i16 {{.*}}, -84
; CHECK-NEXT: ret i16
  %B = getelementptr inbounds [42 x i16], [42 x i16] addrspace(1)* @Arr_as1, i64 0, i64 %A
  %C = ptrtoint i16 addrspace(1)* %B to i16
  %G = sub i16 %C, ptrtoint (i16 addrspace(1)* getelementptr ([42 x i16], [42 x i16] addrspace(1)* @Arr_as1, i64 1, i64 0) to i16)
  ret i16 %G
}

define i32 @test26(i32 %x) {
  %shl = shl i32 3, %x
  %neg = sub i32 0, %shl
  ret i32 %neg
; CHECK-LABEL: @test26(
; CHECK-NEXT: shl i32 -3
; CHECK-NEXT: ret i32
}

define i32 @test27(i32 %x, i32 %y) {
  %mul = mul i32 %y, -8
  %sub = sub i32 %x, %mul
  ret i32 %sub
; CHECK-LABEL: @test27(
; CHECK-NEXT: shl i32 %y, 3
; CHECK-NEXT: add i32
; CHECK-NEXT: ret i32
}

define i32 @test28(i32 %x, i32 %y, i32 %z) {
  %neg = sub i32 0, %z
  %mul = mul i32 %neg, %y
  %sub = sub i32 %x, %mul
  ret i32 %sub
; CHECK-LABEL: @test28(
; CHECK-NEXT: mul i32 %z, %y
; CHECK-NEXT: add i32
; CHECK-NEXT: ret i32
}

define i64 @test29(i8* %foo, i64 %i, i64 %j) {
  %gep1 = getelementptr inbounds i8, i8* %foo, i64 %i
  %gep2 = getelementptr inbounds i8, i8* %foo, i64 %j
  %cast1 = ptrtoint i8* %gep1 to i64
  %cast2 = ptrtoint i8* %gep2 to i64
  %sub = sub i64 %cast1, %cast2
  ret i64 %sub
; CHECK-LABEL: @test29(
; CHECK-NEXT: sub i64 %i, %j
; CHECK-NEXT: ret i64
}

define i64 @test30(i8* %foo, i64 %i, i64 %j) {
  %bit = bitcast i8* %foo to i32*
  %gep1 = getelementptr inbounds i32, i32* %bit, i64 %i
  %gep2 = getelementptr inbounds i8, i8* %foo, i64 %j
  %cast1 = ptrtoint i32* %gep1 to i64
  %cast2 = ptrtoint i8* %gep2 to i64
  %sub = sub i64 %cast1, %cast2
  ret i64 %sub
; CHECK-LABEL: @test30(
; CHECK-NEXT: %gep1.idx = shl nuw i64 %i, 2
; CHECK-NEXT: sub i64 %gep1.idx, %j
; CHECK-NEXT: ret i64
}

define i16 @test30_as1(i8 addrspace(1)* %foo, i16 %i, i16 %j) {
; CHECK-LABEL: @test30_as1(
; CHECK-NEXT: %gep1.idx = shl nuw i16 %i, 2
; CHECK-NEXT: sub i16 %gep1.idx, %j
; CHECK-NEXT: ret i16
  %bit = bitcast i8 addrspace(1)* %foo to i32 addrspace(1)*
  %gep1 = getelementptr inbounds i32, i32 addrspace(1)* %bit, i16 %i
  %gep2 = getelementptr inbounds i8, i8 addrspace(1)* %foo, i16 %j
  %cast1 = ptrtoint i32 addrspace(1)* %gep1 to i16
  %cast2 = ptrtoint i8 addrspace(1)* %gep2 to i16
  %sub = sub i16 %cast1, %cast2
  ret i16 %sub
}

define <2 x i64> @test31(<2 x i64> %A) {
  %xor = xor <2 x i64> %A, <i64 -1, i64 -1>
  %sub = sub <2 x i64> <i64 2, i64 3>, %xor
  ret <2 x i64> %sub
; CHECK-LABEL: @test31(
; CHECK-NEXT: %sub = add <2 x i64> %A, <i64 3, i64 4>
; CHECK-NEXT: ret <2 x i64> %sub
}

define <2 x i64> @test32(<2 x i64> %A) {
  %add = add <2 x i64> %A, <i64 -1, i64 -1>
  %sub = sub <2 x i64> <i64 2, i64 3>, %add
  ret <2 x i64> %sub
; CHECK-LABEL: @test32(
; CHECK-NEXT: %sub = sub <2 x i64> <i64 3, i64 4>
; CHECK-NEXT: ret <2 x i64> %sub
}

define <2 x i64> @test33(<2 x i1> %A) {
  %ext = zext <2 x i1> %A to <2 x i64>
  %sub = sub <2 x i64> zeroinitializer, %ext
  ret <2 x i64> %sub
; CHECK-LABEL: @test33(
; CHECK-NEXT: %sub = sext <2 x i1> %A to <2 x i64>
; CHECK-NEXT: ret <2 x i64> %sub
}

define <2 x i64> @test34(<2 x i1> %A) {
  %ext = sext <2 x i1> %A to <2 x i64>
  %sub = sub <2 x i64> zeroinitializer, %ext
  ret <2 x i64> %sub
; CHECK-LABEL: @test34(
; CHECK-NEXT: %sub = zext <2 x i1> %A to <2 x i64>
; CHECK-NEXT: ret <2 x i64> %sub
}

define <2 x i64> @test35(<2 x i64> %A) {
  %mul = mul <2 x i64> %A, <i64 3, i64 4>
  %sub = sub <2 x i64> %A, %mul
  ret <2 x i64> %sub
; CHECK-LABEL: @test35(
; CHECK-NEXT: %sub = mul <2 x i64> %A, <i64 -2, i64 -3>
; CHECK-NEXT: ret <2 x i64> %sub
}

define <2 x i64> @test36(<2 x i64> %A) {
  %shl = shl <2 x i64> %A, <i64 3, i64 4>
  %sub = sub <2 x i64> %shl, %A
  ret <2 x i64> %sub
; CHECK-LABEL: @test36(
; CHECK-NEXT: %sub = mul <2 x i64> %A, <i64 7, i64 15>
; CHECK-NEXT: ret <2 x i64> %sub
}

define <2 x i32> @test37(<2 x i32> %A) {
  %div = sdiv <2 x i32> %A, <i32 -2147483648, i32 -2147483648>
  %sub = sub nsw <2 x i32> zeroinitializer, %div
  ret <2 x i32> %sub
; CHECK-LABEL: @test37(
; CHECK-NEXT: [[ICMP:%.*]] = icmp eq <2 x i32> %A, <i32 -2147483648, i32 -2147483648>
; CHECK-NEXT: [[SEXT:%.*]] = sext <2 x i1> [[ICMP]] to <2 x i32>
; CHECK-NEXT: ret <2 x i32> [[SEXT]]
}

define i32 @test38(i32 %A) {
  %div = sdiv i32 %A, -2147483648
  %sub = sub nsw i32 0, %div
  ret i32 %sub
; CHECK-LABEL: @test38(
; CHECK-NEXT: [[ICMP:%.*]] = icmp eq i32 %A, -2147483648
; CHECK-NEXT: [[SEXT:%.*]] = sext i1 [[ICMP]] to i32
; CHECK-NEXT: ret i32 [[SEXT]]
}

define i32 @test39(i32 %A, i32 %x) {
  %B = sub i32 0, %A
  %C = sub nsw i32 %x, %B
  ret i32 %C
; CHECK-LABEL: @test39(
; CHECK: %C = add i32 %x, %A
; CHECK: ret i32 %C
}

define i16 @test40(i16 %a, i16 %b) {
  %ashr = ashr i16 %a, 1
  %ashr1 = ashr i16 %b, 1
  %sub = sub i16 %ashr, %ashr1
  ret i16 %sub
; CHECK-LABEL: @test40(
; CHECK-NEXT: [[ASHR:%.*]] = ashr i16 %a, 1
; CHECK-NEXT: [[ASHR1:%.*]] = ashr i16 %b, 1
; CHECK-NEXT: [[RET:%.*]] = sub nsw i16 [[ASHR]], [[ASHR1]]
; CHECK: ret i16 [[RET]]
}

define i32 @test41(i16 %a, i16 %b) {
  %conv = sext i16 %a to i32
  %conv1 = sext i16 %b to i32
  %sub = sub i32 %conv, %conv1
  ret i32 %sub
; CHECK-LABEL: @test41(
; CHECK-NEXT: [[SEXT:%.*]] = sext i16 %a to i32
; CHECK-NEXT: [[SEXT1:%.*]] = sext i16 %b to i32
; CHECK-NEXT: [[RET:%.*]] = sub nsw i32 [[SEXT]], [[SEXT1]]
; CHECK: ret i32 [[RET]]
}

define i4 @test42(i4 %x, i4 %y) {
  %a = and i4 %y, 7
  %b = and i4 %x, 7
  %c = sub i4 %a, %b
  ret i4 %c
; CHECK-LABEL: @test42(
; CHECK-NEXT: [[AND:%.*]] = and i4 %y, 7
; CHECK-NEXT: [[AND1:%.*]] = and i4 %x, 7
; CHECK-NEXT: [[RET:%.*]] = sub nsw i4 [[AND]], [[AND1]]
; CHECK: ret i4 [[RET]]
}

define i4 @test43(i4 %x, i4 %y) {
  %a = or i4 %x, -8
  %b = and i4 %y, 7
  %c = sub i4 %a, %b
  ret i4 %c
; CHECK-LABEL: @test43(
; CHECK-NEXT: [[OR:%.*]] = or i4 %x, -8
; CHECK-NEXT: [[AND:%.*]] = and i4 %y, 7
; CHECK-NEXT: [[RET:%.*]] = sub nuw i4 [[OR]], [[AND]]
; CHECK: ret i4 [[RET]]
}

define i32 @test44(i32 %x) {
  %sub = sub nsw i32 %x, 32768
  ret i32 %sub
; CHECK-LABEL: @test44(
; CHECK-NEXT: [[ADD:%.*]] = add nsw i32 %x, -32768
; CHECK: ret i32 [[ADD]]
}

define i32 @test45(i32 %x, i32 %y) {
  %or = or i32 %x, %y
  %xor = xor i32 %x, %y
  %sub = sub i32 %or, %xor
  ret i32 %sub
; CHECK-LABEL: @test45(
; CHECK-NEXT: %sub = and i32 %x, %y
; CHECK: ret i32 %sub
}

define i32 @test46(i32 %x, i32 %y) {
 %or = or i32 %x, %y
 %sub = sub i32 %or, %x
 ret i32 %sub
; CHECK-LABEL: @test46(
; CHECK-NEXT: %x.not = xor i32 %x, -1
; CHECK-NEXT: %sub = and i32 %y, %x.not
; CHECK: ret i32 %sub
}

define i32 @test47(i1 %A, i32 %B, i32 %C, i32 %D) {
  %sel0 = select i1 %A, i32 %D, i32 %B
  %sel1 = select i1 %A, i32 %C, i32 %B
  %sub = sub i32 %sel0, %sel1
  ret i32 %sub
; CHECK-LABEL: @test47(
; CHECK-NEXT: %[[sub:.*]] = sub i32 %D, %C
; CHECK-NEXT: %[[sel:.*]] = select i1 %A, i32 %[[sub]], i32 0
; CHECK-NEXT: ret i32 %[[sel]]
}

define i32 @test48(i1 %A, i32 %B, i32 %C, i32 %D) {
  %sel0 = select i1 %A, i32 %B, i32 %D
  %sel1 = select i1 %A, i32 %B, i32 %C
  %sub = sub i32 %sel0, %sel1
  ret i32 %sub
; CHECK-LABEL: @test48(
; CHECK-NEXT: %[[sub:.*]] = sub i32 %D, %C
; CHECK-NEXT: %[[sel:.*]] = select i1 %A, i32 0, i32 %[[sub]]
; CHECK-NEXT: ret i32 %[[sel]]
}
