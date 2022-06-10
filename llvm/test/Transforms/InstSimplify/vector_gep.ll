; RUN: opt -S -passes=instsimplify < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

declare void @helper(<2 x ptr>)
define void @test(<2 x ptr> %a) {
  %A = getelementptr i8, <2 x ptr> %a, <2 x i32> <i32 0, i32 0>
  call void @helper(<2 x ptr> %A)
  ret void
}

define <4 x ptr> @test1(<4 x ptr> %a) {
  %gep = getelementptr i8, <4 x ptr> %a, <4 x i32> zeroinitializer
  ret <4 x ptr> %gep

; CHECK-LABEL: @test1
; CHECK-NEXT: ret <4 x ptr> %a
}

define <4 x ptr> @test2(<4 x ptr> %a) {
  %gep = getelementptr i8, <4 x ptr> %a
  ret <4 x ptr> %gep

; CHECK-LABEL: @test2
; CHECK-NEXT: ret <4 x ptr> %a
}

%struct = type { double, float }

define <4 x ptr> @test3() {
  %gep = getelementptr %struct, <4 x ptr> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  ret <4 x ptr> %gep

; CHECK-LABEL: @test3
; CHECK-NEXT: ret <4 x ptr> undef
}

%struct.empty = type { }

define <4 x ptr> @test4(<4 x ptr> %a) {
  %gep = getelementptr %struct.empty, <4 x ptr> %a, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  ret <4 x ptr> %gep

; CHECK-LABEL: @test4
; CHECK-NEXT: ret <4 x ptr> %a
}

define <4 x ptr> @test5() {
  %c = inttoptr <4 x i64> <i64 1, i64 2, i64 3, i64 4> to <4 x ptr>
  %gep = getelementptr i8, <4 x ptr> %c, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  ret <4 x ptr> %gep

; CHECK-LABEL: @test5
; CHECK-NEXT: ret <4 x ptr> getelementptr (i8, <4 x ptr> <ptr inttoptr (i64 1 to ptr), ptr inttoptr (i64 2 to ptr), ptr inttoptr (i64 3 to ptr), ptr inttoptr (i64 4 to ptr)>, <4 x i64> <i64 1, i64 1, i64 1, i64 1>)
}

@v = global [24 x [42 x [3 x i32]]] zeroinitializer, align 16

define <16 x ptr> @test6() {
; CHECK-LABEL: @test6
; CHECK-NEXT: ret <16 x ptr> getelementptr inbounds ([24 x [42 x [3 x i32]]], ptr @v, <16 x i64> zeroinitializer, <16 x i64> zeroinitializer, <16 x i64> <i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, i64 9, i64 10, i64 11, i64 12, i64 13, i64 14, i64 15>, <16 x i64> zeroinitializer)
  %VectorGep = getelementptr [24 x [42 x [3 x i32]]], ptr @v, i64 0, i64 0, <16 x i64> <i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, i64 9, i64 10, i64 11, i64 12, i64 13, i64 14, i64 15>, i64 0
  ret <16 x ptr> %VectorGep
}

; PR32697
; CHECK-LABEL: tinkywinky(
; CHECK-NEXT: ret <4 x ptr> undef
define <4 x ptr> @tinkywinky() {
  %patatino = getelementptr i8, ptr undef, <4 x i64> undef
  ret <4 x ptr> %patatino
}

; PR32697
; CHECK-LABEL: dipsy(
; CHECK-NEXT: ret <4 x ptr> undef
define <4 x ptr> @dipsy() {
  %patatino = getelementptr i8, <4 x ptr> undef, <4 x i64> undef
  ret <4 x ptr> %patatino
}

; PR32697
; CHECK-LABEL: laalaa(
; CHECK-NEXT: ret <4 x ptr> undef
define <4 x ptr> @laalaa() {
  %patatino = getelementptr i8, <4 x ptr> undef, i64 undef
  ret <4 x ptr> %patatino
}

define <2 x ptr> @zero_index(ptr %p) {
; CHECK-LABEL: @zero_index(
; CHECK-NEXT:    %gep = getelementptr i8, ptr %p, <2 x i64> zeroinitializer
; CHECK-NEXT:    ret <2 x ptr> %gep
;
  %gep = getelementptr i8, ptr %p, <2 x i64> zeroinitializer
  ret <2 x ptr> %gep
}

define <2 x ptr> @unsized(ptr %p) {
; CHECK-LABEL: @unsized(
; CHECK-NEXT:    %gep = getelementptr {}, ptr %p, <2 x i64> undef
; CHECK-NEXT:    ret <2 x ptr> %gep
;
  %gep = getelementptr {}, ptr %p, <2 x i64> undef
  ret <2 x ptr> %gep
}
