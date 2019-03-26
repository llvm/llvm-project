; RUN: llc -verify-machineinstrs -o - %s -mtriple=arm64-apple-ios7.0 | FileCheck %s

define i64 @test0() {
; CHECK-LABEL: test0:
; Not produced by move wide instructions, but good to make sure we can return 0 anyway:
; CHECK: mov x0, xzr
  ret i64 0
}

define i64 @test1() {
; CHECK-LABEL: test1:
; CHECK: mov w0, #1
  ret i64 1
}

define i64 @test2() {
; CHECK-LABEL: test2:
; CHECK: mov w0, #65535
  ret i64 65535
}

define i64 @test3() {
; CHECK-LABEL: test3:
; CHECK: mov w0, #65536
  ret i64 65536
}

define i64 @test4() {
; CHECK-LABEL: test4:
; CHECK: mov w0, #-65536
  ret i64 4294901760
}

define i64 @test5() {
; CHECK-LABEL: test5:
; CHECK: mov x0, #4294967296
  ret i64 4294967296
}

define i64 @test6() {
; CHECK-LABEL: test6:
; CHECK: mov x0, #281470681743360
  ret i64 281470681743360
}

define i64 @test7() {
; CHECK-LABEL: test7:
; CHECK: mov x0, #281474976710656
  ret i64 281474976710656
}

; A 32-bit MOVN can generate some 64-bit patterns that a 64-bit one
; couldn't. Useful even for i64
define i64 @test8() {
; CHECK-LABEL: test8:
; CHECK: mov w0, #-60876
  ret i64 4294906420
}

define i64 @test9() {
; CHECK-LABEL: test9:
; CHECK: mov x0, #-1
  ret i64 -1
}

define i64 @test10() {
; CHECK-LABEL: test10:
; CHECK: mov x0, #-3989504001
  ret i64 18446744069720047615
}

; For reasonably legitimate reasons returning an i32 results in the
; selection of an i64 constant, so we need a different idiom to test that selection
@var32 = global i32 0

define void @test11() {
; CHECK-LABEL: test11:
; CHECK: str wzr
  store i32 0, i32* @var32
  ret void
}

define void @test12() {
; CHECK-LABEL: test12:
; CHECK: mov {{w[0-9]+}}, #1
  store i32 1, i32* @var32
  ret void
}

define void @test13() {
; CHECK-LABEL: test13:
; CHECK: mov {{w[0-9]+}}, #65535
  store i32 65535, i32* @var32
  ret void
}

define void @test14() {
; CHECK-LABEL: test14:
; CHECK: mov {{w[0-9]+}}, #65536
  store i32 65536, i32* @var32
  ret void
}

define void @test15() {
; CHECK-LABEL: test15:
; CHECK: mov {{w[0-9]+}}, #-65536
  store i32 4294901760, i32* @var32
  ret void
}

define void @test16() {
; CHECK-LABEL: test16:
; CHECK: mov {{w[0-9]+}}, #-1
  store i32 -1, i32* @var32
  ret void
}

define i64 @test17() {
; CHECK-LABEL: test17:

  ; Mustn't MOVN w0 here.
; CHECK: mov x0, #-3
  ret i64 -3
}
