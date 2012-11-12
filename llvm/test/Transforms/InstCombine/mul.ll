; This test makes sure that mul instructions are properly eliminated.
; RUN: opt < %s -instcombine -S | FileCheck %s

define i32 @test1(i32 %A) {
; CHECK: @test1
        %B = mul i32 %A, 1              ; <i32> [#uses=1]
        ret i32 %B
; CHECK: ret i32 %A
}

define i32 @test2(i32 %A) {
; CHECK: @test2
        ; Should convert to an add instruction
        %B = mul i32 %A, 2              ; <i32> [#uses=1]
        ret i32 %B
; CHECK: shl i32 %A, 1
}

define i32 @test3(i32 %A) {
; CHECK: @test3
        ; This should disappear entirely
        %B = mul i32 %A, 0              ; <i32> [#uses=1]
        ret i32 %B
; CHECK: ret i32 0
}

define double @test4(double %A) {
; CHECK: @test4
        ; This is safe for FP
        %B = fmul double 1.000000e+00, %A                ; <double> [#uses=1]
        ret double %B
; CHECK: ret double %A
}

define i32 @test5(i32 %A) {
; CHECK: @test5
        %B = mul i32 %A, 8              ; <i32> [#uses=1]
        ret i32 %B
; CHECK: shl i32 %A, 3
}

define i8 @test6(i8 %A) {
; CHECK: @test6
        %B = mul i8 %A, 8               ; <i8> [#uses=1]
        %C = mul i8 %B, 8               ; <i8> [#uses=1]
        ret i8 %C
; CHECK: shl i8 %A, 6
}

define i32 @test7(i32 %i) {
; CHECK: @test7
        %tmp = mul i32 %i, -1           ; <i32> [#uses=1]
        ret i32 %tmp
; CHECK: sub i32 0, %i
}

define i64 @test8(i64 %i) {
; CHECK: @test8
        %j = mul i64 %i, -1             ; <i64> [#uses=1]
        ret i64 %j
; CHECK: sub i64 0, %i
}

define i32 @test9(i32 %i) {
; CHECK: @test9
        %j = mul i32 %i, -1             ; <i32> [#uses=1]
        ret i32 %j
; CHECJ: sub i32 0, %i
}

define i32 @test10(i32 %a, i32 %b) {
; CHECK: @test10
        %c = icmp slt i32 %a, 0         ; <i1> [#uses=1]
        %d = zext i1 %c to i32          ; <i32> [#uses=1]
       ; e = b & (a >> 31)
        %e = mul i32 %d, %b             ; <i32> [#uses=1]
        ret i32 %e
; CHECK: [[TEST10:%.*]] = ashr i32 %a, 31
; CHECK-NEXT: %e = and i32 [[TEST10]], %b
; CHECK-NEXT: ret i32 %e
}

define i32 @test11(i32 %a, i32 %b) {
; CHECK: @test11
        %c = icmp sle i32 %a, -1                ; <i1> [#uses=1]
        %d = zext i1 %c to i32          ; <i32> [#uses=1]
        ; e = b & (a >> 31)
        %e = mul i32 %d, %b             ; <i32> [#uses=1]
        ret i32 %e
; CHECK: [[TEST11:%.*]] = ashr i32 %a, 31
; CHECK-NEXT: %e = and i32 [[TEST11]], %b
; CHECK-NEXT: ret i32 %e
}

define i32 @test12(i32 %a, i32 %b) {
; CHECK: @test12
        %c = icmp ugt i32 %a, 2147483647                ; <i1> [#uses=1]
        %d = zext i1 %c to i32          ; <i32> [#uses=1]
        %e = mul i32 %d, %b             ; <i32> [#uses=1]
        ret i32 %e
; CHECK: [[TEST12:%.*]] = ashr i32 %a, 31
; CHECK-NEXT: %e = and i32 [[TEST12]], %b
; CHECK-NEXT: ret i32 %e

}

; PR2642
define internal void @test13(<4 x float>*) {
; CHECK: @test13
	load <4 x float>* %0, align 1
	fmul <4 x float> %2, < float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00 >
	store <4 x float> %3, <4 x float>* %0, align 1
	ret void
; CHECK-NEXT: ret void
}

define <16 x i8> @test14(<16 x i8> %a) {
; CHECK: @test14
        %b = mul <16 x i8> %a, zeroinitializer
        ret <16 x i8> %b
; CHECK-NEXT: ret <16 x i8> zeroinitializer
}

; rdar://7293527
define i32 @test15(i32 %A, i32 %B) {
; CHECK: @test15
entry:
  %shl = shl i32 1, %B
  %m = mul i32 %shl, %A
  ret i32 %m
; CHECK: shl i32 %A, %B
}

; X * Y (when Y is 0 or 1) --> x & (0-Y)
define i32 @test16(i32 %b, i1 %c) {
; CHECK: @test16
        %d = zext i1 %c to i32          ; <i32> [#uses=1]
        ; e = b & (a >> 31)
        %e = mul i32 %d, %b             ; <i32> [#uses=1]
        ret i32 %e
; CHECK: [[TEST16:%.*]] = zext i1 %c to i32
; CHECK-NEXT: %1 = sub i32 0, [[TEST16]]
; CHECK-NEXT: %e = and i32 %1, %b
; CHECK-NEXT: ret i32 %e
}

; X * Y (when Y is 0 or 1) --> x & (0-Y)
define i32 @test17(i32 %a, i32 %b) {
; CHECK: @test17
  %a.lobit = lshr i32 %a, 31
  %e = mul i32 %a.lobit, %b
  ret i32 %e
; CHECK: [[TEST17:%.*]] = ashr i32 %a, 31
; CHECK-NEXT: %e = and i32 [[TEST17]], %b
; CHECK-NEXT: ret i32 %e
}

define i32 @test18(i32 %A, i32 %B) {
; CHECK: @test18
  %C = and i32 %A, 1
  %D = and i32 %B, 1

  %E = mul i32 %C, %D
  %F = and i32 %E, 16
  ret i32 %F
; CHECK-NEXT: ret i32 0
}

declare {i32, i1} @llvm.smul.with.overflow.i32(i32, i32)
declare void @use(i1)

define i32 @test19(i32 %A, i32 %B) {
; CHECK: @test19
  %C = and i32 %A, 1
  %D = and i32 %B, 1

; It would be nice if we also started proving that this doesn't overflow.
  %E = call {i32, i1} @llvm.smul.with.overflow.i32(i32 %C, i32 %D)
  %F = extractvalue {i32, i1} %E, 0
  %G = extractvalue {i32, i1} %E, 1
  call void @use(i1 %G)
  %H = and i32 %F, 16
  ret i32 %H
; CHECK: ret i32 0
}
