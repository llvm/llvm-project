; RUN: opt -instcombine -S < %s | FileCheck %s
; RUN: opt -passes=instcombine -S < %s | FileCheck %s

; This test makes sure that these instructions are properly eliminated.

target datalayout = "e-m:e-p:64:64:64-i64:64-f80:128-n8:16:32:64-S128"

@X = constant i32 42		; <i32*> [#uses=2]
@X2 = constant i32 47		; <i32*> [#uses=1]
@Y = constant [2 x { i32, float }] [ { i32, float } { i32 12, float 1.000000e+00 }, { i32, float } { i32 37, float 0x3FF3B2FEC0000000 } ]		; <[2 x { i32, float }]*> [#uses=2]
@Z = constant [2 x { i32, float }] zeroinitializer		; <[2 x { i32, float }]*> [#uses=1]

@GLOBAL = internal constant [4 x i32] zeroinitializer


; CHECK-LABEL: @test1(
; CHECK-NOT: load
define i32 @test1() {
	%B = load i32, i32* @X		; <i32> [#uses=1]
	ret i32 %B
}

; CHECK-LABEL: @test2(
; CHECK-NOT: load
define float @test2() {
	%A = getelementptr [2 x { i32, float }], [2 x { i32, float }]* @Y, i64 0, i64 1, i32 1		; <float*> [#uses=1]
	%B = load float, float* %A		; <float> [#uses=1]
	ret float %B
}

; CHECK-LABEL: @test3(
; CHECK-NOT: load
define i32 @test3() {
	%A = getelementptr [2 x { i32, float }], [2 x { i32, float }]* @Y, i64 0, i64 0, i32 0		; <i32*> [#uses=1]
	%B = load i32, i32* %A		; <i32> [#uses=1]
	ret i32 %B
}

; CHECK-LABEL: @test4(
; CHECK-NOT: load
define i32 @test4() {
	%A = getelementptr [2 x { i32, float }], [2 x { i32, float }]* @Z, i64 0, i64 1, i32 0		; <i32*> [#uses=1]
	%B = load i32, i32* %A		; <i32> [#uses=1]
	ret i32 %B
}

; CHECK-LABEL: @test5(
; CHECK-NOT: load
define i32 @test5(i1 %C) {
	%Y = select i1 %C, i32* @X, i32* @X2		; <i32*> [#uses=1]
	%Z = load i32, i32* %Y		; <i32> [#uses=1]
	ret i32 %Z
}

; CHECK-LABEL: @test7(
; CHECK-NOT: load
define i32 @test7(i32 %X) {
	%V = getelementptr i32, i32* null, i32 %X		; <i32*> [#uses=1]
	%R = load i32, i32* %V		; <i32> [#uses=1]
	ret i32 %R
}

; CHECK-LABEL: @test8(
; CHECK-NOT: load
define i32 @test8(i32* %P) {
	store i32 1, i32* %P
	%X = load i32, i32* %P		; <i32> [#uses=1]
	ret i32 %X
}

; CHECK-LABEL: @test9(
; CHECK-NOT: load
define i32 @test9(i32* %P) {
	%X = load i32, i32* %P		; <i32> [#uses=1]
	%Y = load i32, i32* %P		; <i32> [#uses=1]
	%Z = sub i32 %X, %Y		; <i32> [#uses=1]
	ret i32 %Z
}

; CHECK-LABEL: @test10(
; CHECK-NOT: load
define i32 @test10(i1 %C.upgrd.1, i32* %P, i32* %Q) {
	br i1 %C.upgrd.1, label %T, label %F
T:		; preds = %0
	store i32 1, i32* %Q
	store i32 0, i32* %P
	br label %C
F:		; preds = %0
	store i32 0, i32* %P
	br label %C
C:		; preds = %F, %T
	%V = load i32, i32* %P		; <i32> [#uses=1]
	ret i32 %V
}

; CHECK-LABEL: @test11(
; CHECK-NOT: load
define double @test11(double* %p) {
  %t0 = getelementptr double, double* %p, i32 1
  store double 2.0, double* %t0
  %t1 = getelementptr double, double* %p, i32 1
  %x = load double, double* %t1
  ret double %x
}

; CHECK-LABEL: @test12(
; CHECK-NOT: load
define i32 @test12(i32* %P) {
  %A = alloca i32
  store i32 123, i32* %A
  ; Cast the result of the load not the source
  %Q = bitcast i32* %A to i32*
  %V = load i32, i32* %Q
  ret i32 %V
}

; CHECK-LABEL: @test13(
; CHECK-NOT: load
define <16 x i8> @test13(<2 x i64> %x) {
  %tmp = load <16 x i8>, <16 x i8>* bitcast ([4 x i32]* @GLOBAL to <16 x i8>*)
  ret <16 x i8> %tmp
}

define i8 @test14(i8 %x, i32 %y) {
; This test must not have the store of %x forwarded to the load -- there is an
; intervening store if %y. However, the intervening store occurs with a different
; type and size and to a different pointer value. This is ensuring that none of
; those confuse the analysis into thinking that the second store does not alias
; the first.
; CHECK-LABEL: @test14(
; CHECK:         %[[R:.*]] = load i8, i8*
; CHECK-NEXT:    ret i8 %[[R]]
  %a = alloca i32
  %a.i8 = bitcast i32* %a to i8*
  store i8 %x, i8* %a.i8
  store i32 %y, i32* %a
  %r = load i8, i8* %a.i8
  ret i8 %r
}

@test15_global = external global i32

define i8 @test15(i8 %x, i32 %y) {
; Same test as @test14 essentially, but using a global instead of an alloca.
; CHECK-LABEL: @test15(
; CHECK:         %[[R:.*]] = load i8, i8*
; CHECK-NEXT:    ret i8 %[[R]]
  %g.i8 = bitcast i32* @test15_global to i8*
  store i8 %x, i8* %g.i8
  store i32 %y, i32* @test15_global
  %r = load i8, i8* %g.i8
  ret i8 %r
}

define void @test16(i8* %x, i8* %a, i8* %b, i8* %c) {
; Check that we canonicalize loads which are only stored to use integer types
; when there is a valid integer type.
; CHECK-LABEL: @test16(
; CHECK: %[[L1:.*]] = load i32, i32*
; CHECK-NOT: load
; CHECK: store i32 %[[L1]], i32*
; CHECK: store i32 %[[L1]], i32*
; CHECK-NOT: store
; CHECK: %[[L1:.*]] = load i32, i32*
; CHECK-NOT: load
; CHECK: store i32 %[[L1]], i32*
; CHECK: store i32 %[[L1]], i32*
; CHECK-NOT: store
; CHECK: ret

entry:
  %x.cast = bitcast i8* %x to float*
  %a.cast = bitcast i8* %a to float*
  %b.cast = bitcast i8* %b to float*
  %c.cast = bitcast i8* %c to i32*

  %x1 = load float, float* %x.cast
  store float %x1, float* %a.cast
  store float %x1, float* %b.cast

  %x2 = load float, float* %x.cast
  store float %x2, float* %b.cast
  %x2.cast = bitcast float %x2 to i32
  store i32 %x2.cast, i32* %c.cast

  ret void
}

define void @test17(i8** %x, i8 %y) {
; Check that in cases similar to @test16 we don't try to rewrite a load when
; its only use is a store but it is used as the pointer to that store rather
; than the value.
;
; CHECK-LABEL: @test17(
; CHECK: %[[L:.*]] = load i8*, i8**
; CHECK: store i8 %y, i8* %[[L]]

entry:
  %x.load = load i8*, i8** %x
  store i8 %y, i8* %x.load

  ret void
}

; Check that we don't try change the type of the load by inserting a bitcast
; generating invalid IR.
; CHECK-LABEL: @test18(
; CHECK-NOT: bitcast
; CHECK: ret
%swift.error = type opaque
declare void @useSwiftError(%swift.error** swifterror)

define void @test18(%swift.error** swifterror %err) {
entry:
  %swifterror = alloca swifterror %swift.error*, align 8
  store %swift.error* null, %swift.error** %swifterror, align 8
  call void @useSwiftError(%swift.error** nonnull swifterror %swifterror)
  %err.res = load %swift.error*, %swift.error** %swifterror, align 8
  store %swift.error* %err.res, %swift.error** %err, align 8
  ret void
}

; Make sure we preseve the type of the store to a swifterror pointer.
; CHECK-LABEL: @test19(
; CHECK: [[A:%.*]] = alloca
; CHECK: call
; CHECK: [[BC:%.*]] = bitcast i8** [[A]] to
; CHECK: [[ERRVAL:%.*]] =  load {{.*}}[[BC]]
; CHECK: store {{.*}}[[ERRVAL]]
; CHECK: ret
declare void @initi8(i8**)
define void @test19(%swift.error** swifterror %err) {
entry:
  %tmp = alloca i8*, align 8
  call void @initi8(i8** %tmp)
  %swifterror = bitcast i8** %tmp to %swift.error**
  %err.res = load %swift.error*, %swift.error** %swifterror, align 8
  store %swift.error* %err.res, %swift.error** %err, align 8
  ret void
}
