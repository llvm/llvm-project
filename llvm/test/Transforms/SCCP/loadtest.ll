; This test makes sure that these instructions are properly constant propagated.

; RUN: opt < %s -data-layout="e-p:32:32" -debugify -sccp -S | FileCheck %s
; RUN: opt < %s -data-layout="E-p:32:32" -debugify -sccp -S | FileCheck %s
; RUN: opt < %s -data-layout="E-p:32:32" -debugify -ipsccp -S | FileCheck %s

@X = constant i32 42		; <i32*> [#uses=1]
@Y = constant [2 x { i32, float }] [ { i32, float } { i32 12, float 1.000000e+00 }, { i32, float } { i32 37, float 0x3FF3B2FEC0000000 } ]		; <[2 x { i32, float }]*> [#uses=2]

define i32 @test1() {
; CHECK-LABEL: @test1(
; CHECK-NEXT: call void @llvm.dbg.value(metadata i32 42
; CHECK-NEXT: ret
	%B = load i32, i32* @X		; <i32> [#uses=1]
	ret i32 %B
}

define float @test2() {
; CHECK-LABEL: @test2(
; CHECK-NEXT: call void @llvm.dbg.value(metadata float* getelementptr
; CHECK-NEXT: call void @llvm.dbg.value(metadata float 0x3FF3B2FEC0000000
; CHECK-NEXT: ret
	%A = getelementptr [2 x { i32, float }], [2 x { i32, float }]* @Y, i64 0, i64 1, i32 1		; <float*> [#uses=1]
	%B = load float, float* %A		; <float> [#uses=1]
	ret float %B
}

define i32 @test3() {
; CHECK-LABEL: @test3(
; CHECK-NEXT: call void @llvm.dbg.value(metadata i32* getelementptr
; CHECK-NEXT: call void @llvm.dbg.value(metadata i32 12
; CHECK-NEXT: ret
	%A = getelementptr [2 x { i32, float }], [2 x { i32, float }]* @Y, i64 0, i64 0, i32 0		; <i32*> [#uses=1]
	%B = load i32, i32* %A
	ret i32 %B
}

define i8 @test4() {
; CHECK-LABEL: @test4(
; CHECK-NEXT: call void @llvm.dbg.value(metadata i8* bitcast
; CHECK-NEXT: call void @llvm.dbg.value(metadata i8
; CHECK-NEXT: ret
	%A = bitcast i32* @X to i8*
	%B = load i8, i8* %A
	ret i8 %B
}
