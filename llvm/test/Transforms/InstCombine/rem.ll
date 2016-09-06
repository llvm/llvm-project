; This test makes sure that rem instructions are properly eliminated.
;
; RUN: opt < %s -instcombine -S | FileCheck %s
; END.

define i32 @test1(i32 %A) {
; CHECK-LABEL: @test1(
; CHECK-NEXT: ret i32 0
	%B = srem i32 %A, 1	; ISA constant 0
	ret i32 %B
}

define i32 @test2(i32 %A) {	; 0 % X = 0, we don't need to preserve traps
; CHECK-LABEL: @test2(
; CHECK-NEXT: ret i32 0
	%B = srem i32 0, %A
	ret i32 %B
}

define i32 @test3(i32 %A) {
; CHECK-LABEL: @test3(
; CHECK-NEXT: [[AND:%.*]] = and i32 %A, 7
; CHECK-NEXT: ret i32 [[AND]]
	%B = urem i32 %A, 8
	ret i32 %B
}

define <2 x i32> @vec_power_of_2_constant_splat_divisor(<2 x i32> %A) {
; CHECK-LABEL: @vec_power_of_2_constant_splat_divisor(
; CHECK-NEXT:    [[B:%.*]] = and <2 x i32> %A, <i32 7, i32 7>
; CHECK-NEXT:    ret <2 x i32> [[B]]
;
  %B = urem <2 x i32> %A, <i32 8, i32 8>
  ret <2 x i32> %B
}

define <2 x i19> @weird_vec_power_of_2_constant_splat_divisor(<2 x i19> %A) {
; CHECK-LABEL: @weird_vec_power_of_2_constant_splat_divisor(
; CHECK-NEXT:    [[B:%.*]] = and <2 x i19> %A, <i19 7, i19 7>
; CHECK-NEXT:    ret <2 x i19> [[B]]
;
  %B = urem <2 x i19> %A, <i19 8, i19 8>
  ret <2 x i19> %B
}

define i1 @test3a(i32 %A) {
; CHECK-LABEL: @test3a(
; CHECK-NEXT: [[AND:%.*]] = and i32 %A, 7
; CHECK-NEXT: [[CMP:%.*]] = icmp ne i32 [[AND]], 0
; CHECK-NEXT: ret i1 [[CMP]]
	%B = srem i32 %A, -8
	%C = icmp ne i32 %B, 0
	ret i1 %C
}

define <2 x i1> @test3a_vec(<2 x i32> %A) {
; CHECK-LABEL: @test3a_vec(
; CHECK-NEXT:    [[B1:%.*]] = and <2 x i32> %A, <i32 7, i32 7>
; CHECK-NEXT:    [[C:%.*]] = icmp ne <2 x i32> [[B1]], zeroinitializer
; CHECK-NEXT:    ret <2 x i1> [[C]]
;
  %B = srem <2 x i32> %A, <i32 -8, i32 -8>
  %C = icmp ne <2 x i32> %B, zeroinitializer
  ret <2 x i1> %C
}

define i32 @test4(i32 %X, i1 %C) {
; CHECK-LABEL: @test4(
; CHECK-NEXT: [[SEL:%.*]] = select i1 %C, i32 0, i32 7
; CHECK-NEXT: [[AND:%.*]] = and i32 [[SEL]], %X
	%V = select i1 %C, i32 1, i32 8
	%R = urem i32 %X, %V
	ret i32 %R
}

define i32 @test5(i32 %X, i8 %B) {
; CHECK-LABEL: @test5(
; CHECK-NEXT: [[ZEXT:%.*]] = zext i8 %B to i32
; CHECK-NEXT: [[SHL:%.*]] = shl nuw i32 32, [[ZEXT]]
; CHECK-NEXT: [[ADD:%.*]] = add i32 [[SHL]], -1
; CHECK-NEXT: [[AND:%.*]] = and i32 [[ADD]], %X
; CHECK-NEXT: ret i32 [[AND]]
	%shift.upgrd.1 = zext i8 %B to i32
	%Amt = shl i32 32, %shift.upgrd.1
	%V = urem i32 %X, %Amt
	ret i32 %V
}

define i32 @test6(i32 %A) {
; CHECK-LABEL: @test6(
; CHECK-NEXT: ret i32 undef
	%B = srem i32 %A, 0	;; undef
	ret i32 %B
}

define i32 @test7(i32 %A) {
; CHECK-LABEL: @test7(
; CHECK-NEXT: ret i32 0
	%B = mul i32 %A, 8
	%C = srem i32 %B, 4
	ret i32 %C
}

define i32 @test8(i32 %A) {
; CHECK-LABEL: @test8(
; CHECK-NEXT: ret i32 0
	%B = shl i32 %A, 4
	%C = srem i32 %B, 8
	ret i32 %C
}

define i32 @test9(i32 %A) {
; CHECK-LABEL: @test9(
; CHECK-NEXT: ret i32 0
	%B = mul i32 %A, 64
	%C = urem i32 %B, 32
	ret i32 %C
}

define i32 @test10(i8 %c) {
; CHECK-LABEL: @test10(
; CHECK-NEXT: ret i32 0
	%tmp.1 = zext i8 %c to i32
	%tmp.2 = mul i32 %tmp.1, 4
	%tmp.3 = sext i32 %tmp.2 to i64
	%tmp.5 = urem i64 %tmp.3, 4
	%tmp.6 = trunc i64 %tmp.5 to i32
	ret i32 %tmp.6
}

define i32 @test11(i32 %i) {
; CHECK-LABEL: @test11(
; CHECK-NEXT: ret i32 0
	%tmp.1 = and i32 %i, -2
	%tmp.3 = mul i32 %tmp.1, 2
	%tmp.5 = urem i32 %tmp.3, 4
	ret i32 %tmp.5
}

define i32 @test12(i32 %i) {
; CHECK-LABEL: @test12(
; CHECK-NEXT: ret i32 0
	%tmp.1 = and i32 %i, -4
	%tmp.5 = srem i32 %tmp.1, 2
	ret i32 %tmp.5
}

define i32 @test13(i32 %i) {
; CHECK-LABEL: @test13(
; CHECK-NEXT: ret i32 0
	%x = srem i32 %i, %i
	ret i32 %x
}

define i64 @test14(i64 %x, i32 %y) {
; CHECK-LABEL: @test14(
; CHECK-NEXT: [[SHL:%.*]] = shl i32 1, %y
; CHECK-NEXT: [[ZEXT:%.*]] = zext i32 [[SHL]] to i64
; CHECK-NEXT: [[ADD:%.*]] = add nsw i64 [[ZEXT]], -1
; CHECK-NEXT: [[AND:%.*]] = and i64 [[ADD]], %x
; CHECK-NEXT: ret i64 [[AND]]
	%shl = shl i32 1, %y
	%zext = zext i32 %shl to i64
	%urem = urem i64 %x, %zext
	ret i64 %urem
}

define i64 @test15(i32 %x, i32 %y) {
; CHECK-LABEL: @test15(
; CHECK-NEXT: [[SHL:%.*]] = shl nuw i32 1, %y
; CHECK-NEXT: [[ADD:%.*]] = add i32 [[SHL]], -1
; CHECK-NEXT: [[AND:%.*]] = and i32 [[ADD]], %x
; CHECK-NEXT: [[ZEXT:%.*]] = zext i32 [[AND]] to i64
; CHECK-NEXT: ret i64 [[ZEXT]]
	%shl = shl i32 1, %y
	%zext0 = zext i32 %shl to i64
	%zext1 = zext i32 %x to i64
	%urem = urem i64 %zext1, %zext0
	ret i64 %urem
}

define i32 @test16(i32 %x, i32 %y) {
; CHECK-LABEL: @test16(
; CHECK-NEXT: [[SHR:%.*]] = lshr i32 %y, 11
; CHECK-NEXT: [[AND:%.*]] = and i32 [[SHR]], 4
; CHECK-NEXT: [[OR:%.*]] = or i32 [[AND]], 3
; CHECK-NEXT: [[REM:%.*]] = and i32 [[OR]], %x
; CHECK-NEXT: ret i32 [[REM]]
	%shr = lshr i32 %y, 11
	%and = and i32 %shr, 4
	%add = add i32 %and, 4
	%rem = urem i32 %x, %add
	ret i32 %rem
}

define i32 @test17(i32 %X) {
; CHECK-LABEL: @test17(
; CHECK-NEXT: icmp ne i32 %X, 1
; CHECK-NEXT: zext i1
; CHECK-NEXT: ret
  %A = urem i32 1, %X
  ret i32 %A
}

define i32 @test18(i16 %x, i32 %y) {
; CHECK: @test18
; CHECK-NEXT: [[AND:%.*]] = and i16 %x, 4
; CHECK-NEXT: [[EXT:%.*]] = zext i16 [[AND]] to i32
; CHECK-NEXT: [[SHL:%.*]] = shl nuw nsw i32 [[EXT]], 3
; CHECK-NEXT: [[XOR:%.*]] = xor i32 [[SHL]], 63
; CHECK-NEXT: [[REM:%.*]] = and i32 [[XOR]], %y
; CHECK-NEXT: ret i32 [[REM]]
	%1 = and i16 %x, 4
	%2 = icmp ne i16 %1, 0
	%3 = select i1 %2, i32 32, i32 64
	%4 = urem i32 %y, %3
	ret i32 %4
}

define i32 @test19(i32 %x, i32 %y) {
; CHECK: @test19
; CHECK-NEXT: [[SHL1:%.*]] = shl i32 1, %x
; CHECK-NEXT: [[SHL2:%.*]] = shl i32 1, %y
; CHECK-NEXT: [[AND:%.*]] = and i32 [[SHL1]], [[SHL2]]
; CHECK-NEXT: [[ADD:%.*]] = add i32 [[AND]], [[SHL1]]
; CHECK-NEXT: [[SUB:%.*]] = add i32 [[ADD]], -1
; CHECK-NEXT: [[REM:%.*]] = and i32 [[SUB]], %y
; CHECK-NEXT: ret i32 [[REM]]
	%A = shl i32 1, %x
	%B = shl i32 1, %y
	%C = and i32 %A, %B
	%D = add i32 %C, %A
	%E = urem i32 %y, %D
	ret i32 %E
}

define <2 x i64> @test20(<2 x i64> %X, <2 x i1> %C) {
; CHECK-LABEL: @test20(
; CHECK-NEXT: select <2 x i1> %C, <2 x i64> <i64 1, i64 2>, <2 x i64> zeroinitializer
; CHECK-NEXT: ret <2 x i64>
	%V = select <2 x i1> %C, <2 x i64> <i64 1, i64 2>, <2 x i64> <i64 8, i64 9>
	%R = urem <2 x i64> %V, <i64 2, i64 3>
	ret <2 x i64> %R
}

define i32 @test21(i1 %c0, i32* %val) {
; CHECK-LABEL: @test21(
entry:
  br i1 %c0, label %if.then, label %if.end

if.then:
; CHECK: if.then:
; CHECK-NEXT:  %v = load volatile i32, i32* %val, align 4
; CHECK-NEXT:  %phitmp = srem i32 %v, 5

  %v = load volatile i32, i32* %val
  br label %if.end

if.end:
; CHECK: if.end:
; CHECK-NEXT:  %lhs = phi i32 [ %phitmp, %if.then ], [ 0, %entry ]
; CHECK-NEXT:  ret i32 %lhs

  %lhs = phi i32 [ %v, %if.then ], [ 5, %entry ]
  %rem = srem i32 %lhs, 5
  ret i32 %rem
}

@a = common global [5 x i16] zeroinitializer, align 2
@b = common global i16 0, align 2

define i32 @pr27968_0(i1 %c0, i32* %val) {
; CHECK-LABEL: @pr27968_0(
entry:
  br i1 %c0, label %if.then, label %if.end

if.then:
  %v = load volatile i32, i32* %val
  br label %if.end

; CHECK: if.then:
; CHECK-NOT: srem
; CHECK:  br label %if.end

if.end:
  %lhs = phi i32 [ %v, %if.then ], [ 5, %entry ]
  br i1 icmp eq (i16* getelementptr inbounds ([5 x i16], [5 x i16]* @a, i64 0, i64 4), i16* @b), label %rem.is.safe, label %rem.is.unsafe

rem.is.safe:
; CHECK: rem.is.safe:
; CHECK-NEXT:  %rem = srem i32 %lhs, zext (i1 icmp eq (i16* getelementptr inbounds ([5 x i16], [5 x i16]* @a, i64 0, i64 4), i16* @b) to i32)
; CHECK-NEXT:  ret i32 %rem

  %rem = srem i32 %lhs, zext (i1 icmp eq (i16* getelementptr inbounds ([5 x i16], [5 x i16]* @a, i64 0, i64 4), i16* @b) to i32)
  ret i32 %rem

rem.is.unsafe:
  ret i32 0
}

define i32 @pr27968_1(i1 %c0, i1 %always_false, i32* %val) {
; CHECK-LABEL: @pr27968_1(
entry:
  br i1 %c0, label %if.then, label %if.end

if.then:
  %v = load volatile i32, i32* %val
  br label %if.end

; CHECK: if.then:
; CHECK-NOT: srem
; CHECK:  br label %if.end

if.end:
  %lhs = phi i32 [ %v, %if.then ], [ 5, %entry ]
  br i1 %always_false, label %rem.is.safe, label %rem.is.unsafe

rem.is.safe:
  %rem = srem i32 %lhs, -2147483648
  ret i32 %rem

; CHECK: rem.is.safe:
; CHECK-NEXT:  %rem = srem i32 %lhs, -2147483648
; CHECK-NEXT:  ret i32 %rem

rem.is.unsafe:
  ret i32 0
}

define i32 @pr27968_2(i1 %c0, i32* %val) {
; CHECK-LABEL: @pr27968_2(
entry:
  br i1 %c0, label %if.then, label %if.end

if.then:
  %v = load volatile i32, i32* %val
  br label %if.end

; CHECK: if.then:
; CHECK-NOT: urem
; CHECK:  br label %if.end

if.end:
  %lhs = phi i32 [ %v, %if.then ], [ 5, %entry ]
  br i1 icmp eq (i16* getelementptr inbounds ([5 x i16], [5 x i16]* @a, i64 0, i64 4), i16* @b), label %rem.is.safe, label %rem.is.unsafe

rem.is.safe:
; CHECK: rem.is.safe:
; CHECK-NEXT:  %rem = urem i32 %lhs, zext (i1 icmp eq (i16* getelementptr inbounds ([5 x i16], [5 x i16]* @a, i64 0, i64 4), i16* @b) to i32)
; CHECK-NEXT:  ret i32 %rem

  %rem = urem i32 %lhs, zext (i1 icmp eq (i16* getelementptr inbounds ([5 x i16], [5 x i16]* @a, i64 0, i64 4), i16* @b) to i32)
  ret i32 %rem

rem.is.unsafe:
  ret i32 0
}

define i32 @pr27968_3(i1 %c0, i1 %always_false, i32* %val) {
; CHECK-LABEL: @pr27968_3(
entry:
  br i1 %c0, label %if.then, label %if.end

if.then:
  %v = load volatile i32, i32* %val
  br label %if.end

; CHECK: if.then:
; CHECK-NEXT:  %v = load volatile i32, i32* %val, align 4
; CHECK-NEXT:  %phitmp = and i32 %v, 2147483647
; CHECK-NEXT:  br label %if.end

if.end:
  %lhs = phi i32 [ %v, %if.then ], [ 5, %entry ]
  br i1 %always_false, label %rem.is.safe, label %rem.is.unsafe

rem.is.safe:
  %rem = urem i32 %lhs, -2147483648
  ret i32 %rem

rem.is.unsafe:
  ret i32 0
}
