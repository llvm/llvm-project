; RUN: opt -instcombine -S -o - %s | FileCheck %s
; Tests that we don't crash upon encountering a vector GEP

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%Dual = type { %Dual.72, %Partials.73 }
%Dual.72 = type { double, %Partials }
%Partials = type { [2 x double] }
%Partials.73 = type { [2 x %Dual.72] }

; Function Attrs: sspreq
define <8 x ptr> @"julia_axpy!_65480"(ptr %arg1, <8 x i64> %arg2) {
top:
; CHECK: %VectorGep14 = getelementptr inbounds %Dual, ptr %arg1, <8 x i64> %arg2, i32 1, i32 0, i64 0, i32 1, i32 0, i64 0
  %VectorGep14 = getelementptr inbounds %Dual, ptr %arg1, <8 x i64> %arg2, i32 1, i32 0, i64 0, i32 1, i32 0, i64 0
  %0 = bitcast <8 x ptr> %VectorGep14 to <8 x ptr>
  ret <8 x ptr> %0
}

%struct.A = type { i32, ptr }
%struct.B = type { i64, ptr }
%struct.C = type { i64 }

@G = internal global [65 x %struct.A] zeroinitializer, align 16
; CHECK-LABEL: @test
; CHECK: ret <16 x ptr> getelementptr ([65 x %struct.A], ptr @G, <16 x i64> zeroinitializer, <16 x i64> <i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, i64 9, i64 10, i64 11, i64 12, i64 13, i64 14, i64 15, i64 16>, i32 0)
define <16 x ptr> @test() {
vector.body:
  %VectorGep = getelementptr [65 x %struct.A], ptr @G, <16 x i64> zeroinitializer, <16 x i64> <i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, i64 9, i64 10, i64 11, i64 12, i64 13, i64 14, i64 15, i64 16>, <16 x i32> zeroinitializer
  ret <16 x ptr> %VectorGep
}

; CHECK-LABEL: @test2
; CHECK: ret <16 x ptr> getelementptr ([65 x %struct.A], ptr @G, <16 x i64> zeroinitializer, <16 x i64> <i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, i64 9, 
define <16 x ptr> @test2() {
vector.body:
  %VectorGep = getelementptr [65 x %struct.A], ptr @G, <16 x i32> zeroinitializer, <16 x i64> <i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, i64 9, i64 10, i64 11, i64 12, i64 13, i64 14, i64 15, i64 16>, <16 x i32> zeroinitializer
  ret <16 x ptr> %VectorGep
}

@g = external global i8, align 1

define <2 x ptr> @constant_zero_index() {
; CHECK-LABEL: @constant_zero_index(
; CHECK-NEXT:    ret <2 x ptr> <ptr @g, ptr @g>
;
  %gep = getelementptr i8, ptr @g, <2 x i64> zeroinitializer
  ret <2 x ptr> %gep
}

define <2 x ptr> @constant_undef_index() {
; CHECK-LABEL: @constant_undef_index(
; CHECK-NEXT:    ret <2 x ptr> <ptr @g, ptr @g>
;
  %gep = getelementptr i8, ptr @g, <2 x i64> undef
  ret <2 x ptr> %gep
}

define <2 x ptr> @constant_inbounds() {
; CHECK-LABEL: @constant_inbounds(
; CHECK-NEXT:    ret <2 x ptr> getelementptr inbounds (i8, ptr @g, <2 x i64> <i64 1, i64 1>)
;
  %gep = getelementptr i8, ptr @g, <2 x i64> <i64 1, i64 1>
  ret <2 x ptr> %gep
}
