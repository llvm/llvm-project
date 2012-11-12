; Test ensures that non-optimizable array mallocs are not optimized; specifically
; GlobalOpt was treating a non-optimizable array malloc as a non-array malloc
; and optimizing the global object that the malloc was stored to as a single
; element global.  The global object @TOP in this test should not be optimized.
; RUN: opt < %s -globalopt -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-apple-darwin10.0"

@TOP = internal global i64* null                    ; <i64**> [#uses=2]
; CHECK: @TOP = internal unnamed_addr global i64* null
@channelColumns = internal global i64 0             ; <i64*> [#uses=2]

; Derived from @DescribeChannel() in yacr2
define void @test() nounwind ssp {
  store i64 2335, i64* @channelColumns, align 8
  %1 = load i64* @channelColumns, align 8         ; <i64> [#uses=1]
  %2 = shl i64 %1, 3                              ; <i64> [#uses=1]
  %3 = add i64 %2, 8                              ; <i64> [#uses=1]
  %4 = call noalias i8* @malloc(i64 %3) nounwind  ; <i8*> [#uses=1]
; CHECK: call noalias i8* @malloc
  %5 = bitcast i8* %4 to i64*                     ; <i64*> [#uses=1]
  store i64* %5, i64** @TOP, align 8
  %6 = load i64** @TOP, align 8                   ; <i64*> [#uses=1]
  %7 = getelementptr inbounds i64* %6, i64 13     ; <i64*> [#uses=1]
  store i64 0, i64* %7, align 8
  ret void
}

declare noalias i8* @malloc(i64) nounwind
