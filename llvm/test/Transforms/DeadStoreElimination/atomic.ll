; RUN: opt -basic-aa -dse -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.7.0"

; Sanity tests for atomic stores.
; Note that it turns out essentially every transformation DSE does is legal on
; atomic ops, just some transformations are not allowed across release-acquire pairs.

@x = common global i32 0, align 4
@y = common global i32 0, align 4

declare void @randomop(i32*)

; DSE across unordered store (allowed)
define void @test1() {
; CHECK-LABEL: test1
; CHECK-NOT: store i32 0
; CHECK: store i32 1
  store i32 0, i32* @x
  store atomic i32 0, i32* @y unordered, align 4
  store i32 1, i32* @x
  ret void
}

; DSE remove unordered store (allowed)
define void @test4() {
; CHECK-LABEL: test4
; CHECK-NOT: store atomic
; CHECK: store i32 1
  store atomic i32 0, i32* @x unordered, align 4
  store i32 1, i32* @x
  ret void
}

; DSE unordered store overwriting non-atomic store (allowed)
define void @test5() {
; CHECK-LABEL: test5
; CHECK: store atomic i32 1
  store i32 0, i32* @x
  store atomic i32 1, i32* @x unordered, align 4
  ret void
}

; DSE no-op unordered atomic store (allowed)
define void @test6() {
; CHECK-LABEL: test6
; CHECK-NOT: store
; CHECK: ret void
  %x = load atomic i32, i32* @x unordered, align 4
  store atomic i32 %x, i32* @x unordered, align 4
  ret void
}

; DSE seq_cst store (be conservative; DSE doesn't have infrastructure
; to reason about atomic operations).
define void @test7() {
; CHECK-LABEL: test7
; CHECK: store atomic
  %a = alloca i32
  store atomic i32 0, i32* %a seq_cst, align 4
  ret void
}

; DSE and seq_cst load (be conservative; DSE doesn't have infrastructure
; to reason about atomic operations).
define i32 @test8() {
; CHECK-LABEL: test8
; CHECK: store
; CHECK: load atomic
  %a = alloca i32
  call void @randomop(i32* %a)
  store i32 0, i32* %a, align 4
  %x = load atomic i32, i32* @x seq_cst, align 4
  ret i32 %x
}

; DSE across monotonic load (allowed as long as the eliminated store isUnordered)
define i32 @test9() {
; CHECK-LABEL: test9
; CHECK-NOT: store i32 0
; CHECK: store i32 1
  store i32 0, i32* @x
  %x = load atomic i32, i32* @y monotonic, align 4
  store i32 1, i32* @x
  ret i32 %x
}

; DSE across monotonic store (allowed as long as the eliminated store isUnordered)
define void @test10() {
; CHECK-LABEL: test10
; CHECK-NOT: store i32 0
; CHECK: store i32 1
  store i32 0, i32* @x
  store atomic i32 42, i32* @y monotonic, align 4
  store i32 1, i32* @x
  ret void
}

; DSE across monotonic load (forbidden since the eliminated store is atomic)
define i32 @test11() {
; CHECK-LABEL: test11
; CHECK: store atomic i32 0
; CHECK: store atomic i32 1
  store atomic i32 0, i32* @x monotonic, align 4
  %x = load atomic i32, i32* @y monotonic, align 4
  store atomic i32 1, i32* @x monotonic, align 4
  ret i32 %x
}

; DSE across monotonic store (forbidden since the eliminated store is atomic)
define void @test12() {
; CHECK-LABEL: test12
; CHECK: store atomic i32 0
; CHECK: store atomic i32 1
  store atomic i32 0, i32* @x monotonic, align 4
  store atomic i32 42, i32* @y monotonic, align 4
  store atomic i32 1, i32* @x monotonic, align 4
  ret void
}

; But DSE is not allowed across a release-acquire pair.
define i32 @test15() {
; CHECK-LABEL: test15
; CHECK: store i32 0
; CHECK: store i32 1
  store i32 0, i32* @x
  store atomic i32 0, i32* @y release, align 4
  %x = load atomic i32, i32* @y acquire, align 4
  store i32 1, i32* @x
  ret i32 %x
}
