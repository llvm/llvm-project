; RUN: opt -S -passes=dse < %s | FileCheck %s

declare void @f()
declare noalias ptr @malloc(i32) nounwind

define void @test_0() {
; CHECK-LABEL: @test_0(
  %m = call ptr @malloc(i32 24)
  tail call void @f() [ "unknown"(ptr %m) ]
; CHECK: store i8 -19, ptr %m
  store i8 -19, ptr %m
  ret void
}

define ptr @test_1() {
; CHECK-LABEL: @test_1(
  %m = call ptr @malloc(i32 24)
  tail call void @f() [ "unknown"(ptr %m) ]
  store i8 -19, ptr %m
  tail call void @f()
  store i8 101, ptr %m

; CHECK: tail call void @f() [ "unknown"(ptr %m) ]
; CHECK: store i8 -19, ptr %m
; CHECK: tail call void @f()
; CHECK: store i8 101, ptr %m

  ret ptr %m
}

define void @test_2() {
; Since the deopt operand bundle does not escape %m (see caveat below), it is
; legal to elide the final store that location.

; CHECK-LABEL: @test_2(
  %m = call ptr @malloc(i32 24)
  tail call void @f() [ "deopt"(ptr %m) ]
  store i8 -19, ptr %m
  ret void

; CHECK:  tail call void @f() [ "deopt"(ptr %m) ]
; CHECK-NEXT:  ret void
}

define ptr @test_3() {
; Since the deopt operand bundle does not escape %m (see caveat below), @f
; cannot observe the stores to %m

; CHECK-LABEL: @test_3(
  %m = call ptr @malloc(i32 24)
  tail call void @f() [ "deopt"(ptr %m) ]
  store i8 -19, ptr %m
  tail call void @f()
  store i8 101, ptr %m
  ret ptr %m
}


; Caveat: technically, %m can only escape if the calling function is deoptimized
; at the call site (i.e. the call returns to the "deopt" continuation).  Since
; the calling function body will be invalidated in that case, the calling
; function can be optimized under the assumption that %m does not escape.
