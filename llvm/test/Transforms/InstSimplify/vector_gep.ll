; RUN: opt -S -instsimplify < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

declare void @helper(<2 x i8*>)
define void @test(<2 x i8*> %a) {
  %A = getelementptr i8, <2 x i8*> %a, <2 x i32> <i32 0, i32 0>
  call void @helper(<2 x i8*> %A)
  ret void
}

define <4 x i8*> @test1(<4 x i8*> %a) {
  %gep = getelementptr i8, <4 x i8*> %a, <4 x i32> zeroinitializer
  ret <4 x i8*> %gep

; CHECK-LABEL: @test1
; CHECK-NEXT: ret <4 x i8*> %a
}

define <4 x i8*> @test2(<4 x i8*> %a) {
  %gep = getelementptr i8, <4 x i8*> %a
  ret <4 x i8*> %gep

; CHECK-LABEL: @test2
; CHECK-NEXT: ret <4 x i8*> %a
}

%struct = type { double, float }

define <4 x float*> @test3() {
  %gep = getelementptr %struct, <4 x %struct*> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  ret <4 x float*> %gep

; CHECK-LABEL: @test3
; CHECK-NEXT: ret <4 x float*> undef
}

%struct.empty = type { }

define <4 x %struct.empty*> @test4(<4 x %struct.empty*> %a) {
  %gep = getelementptr %struct.empty, <4 x %struct.empty*> %a, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  ret <4 x %struct.empty*> %gep

; CHECK-LABEL: @test4
; CHECK-NEXT: ret <4 x %struct.empty*> %a
}

define <4 x i8*> @test5() {
  %c = inttoptr <4 x i64> <i64 1, i64 2, i64 3, i64 4> to <4 x i8*>
  %gep = getelementptr i8, <4 x i8*> %c, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  ret <4 x i8*> %gep

; CHECK-LABEL: @test5
; CHECK-NEXT: ret <4 x i8*> getelementptr (i8, <4 x i8*> <i8* inttoptr (i64 1 to i8*), i8* inttoptr (i64 2 to i8*), i8* inttoptr (i64 3 to i8*), i8* inttoptr (i64 4 to i8*)>, <4 x i64> <i64 1, i64 1, i64 1, i64 1>)
}

@v = global [24 x [42 x [3 x i32]]] zeroinitializer, align 16

define <16 x i32*> @test6() {
; CHECK-LABEL: @test6
; CHECK-NEXT: ret <16 x i32*> getelementptr ([24 x [42 x [3 x i32]]], [24 x [42 x [3 x i32]]]* @v, <16 x i64> zeroinitializer, <16 x i64> zeroinitializer, <16 x i64> <i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, i64 9, i64 10, i64 11, i64 12, i64 13, i64 14, i64 15>, <16 x i64> zeroinitializer)
  %VectorGep = getelementptr [24 x [42 x [3 x i32]]], [24 x [42 x [3 x i32]]]* @v, i64 0, i64 0, <16 x i64> <i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, i64 9, i64 10, i64 11, i64 12, i64 13, i64 14, i64 15>, i64 0
  ret <16 x i32*> %VectorGep
}

; PR32697
; CHECK-LABEL: tinkywinky(
; CHECK-NEXT: ret <4 x i8*> undef
define <4 x i8*> @tinkywinky() {
  %patatino = getelementptr i8, i8* undef, <4 x i64> undef
  ret <4 x i8*> %patatino
}

; PR32697
; CHECK-LABEL: dipsy(
; CHECK-NEXT: ret <4 x i8*> undef
define <4 x i8*> @dipsy() {
  %patatino = getelementptr i8, <4 x i8 *> undef, <4 x i64> undef
  ret <4 x i8*> %patatino
}

; PR32697
; CHECK-LABEL: laalaa(
; CHECK-NEXT: ret <4 x i8*> undef
define <4 x i8*> @laalaa() {
  %patatino = getelementptr i8, <4 x i8 *> undef, i64 undef
  ret <4 x i8*> %patatino
}
