; RUN: llc -mtriple=arm64-linux-gnu -fast-isel=0 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=arm64-linux-gnu -fast-isel=1 -verify-machineinstrs < %s | FileCheck %s

define i32 @test1(i32 %x) {
; CHECK-LABEL: test1
; CHECK: add w8, w0, #7
; CHECK: cmp w0, #0
; CHECK: csel w8, w8, w0, lt
; CHECK: asr w0, w8, #3
  %div = sdiv i32 %x, 8
  ret i32 %div
}

define i32 @test2(i32 %x) {
; CHECK-LABEL: test2
; CHECK: add w8, w0, #7
; CHECK: cmp w0, #0
; CHECK: csel w8, w8, w0, lt
; CHECK: neg w0, w8, asr #3
  %div = sdiv i32 %x, -8
  ret i32 %div
}

define i32 @test3(i32 %x) {
; CHECK-LABEL: test3
; CHECK: add w8, w0, #31
; CHECK: cmp w0, #0
; CHECK: csel w8, w8, w0, lt
; CHECK: asr w0, w8, #5
  %div = sdiv i32 %x, 32
  ret i32 %div
}

define i64 @test4(i64 %x) {
; CHECK-LABEL: test4
; CHECK: add x8, x0, #7
; CHECK: cmp x0, #0
; CHECK: csel x8, x8, x0, lt
; CHECK: asr x0, x8, #3
  %div = sdiv i64 %x, 8
  ret i64 %div
}

define i64 @test5(i64 %x) {
; CHECK-LABEL: test5
; CHECK: add x8, x0, #7
; CHECK: cmp x0, #0
; CHECK: csel x8, x8, x0, lt
; CHECK: neg x0, x8, asr #3
  %div = sdiv i64 %x, -8
  ret i64 %div
}

define i64 @test6(i64 %x) {
; CHECK-LABEL: test6
; CHECK: add x8, x0, #63
; CHECK: cmp x0, #0
; CHECK: csel x8, x8, x0, lt
; CHECK: asr x0, x8, #6
  %div = sdiv i64 %x, 64
  ret i64 %div
}

define i64 @test7(i64 %x) {
; CHECK-LABEL: test7
; CHECK: orr [[REG:x[0-9]+]], xzr, #0xffffffffffff
; CHECK: add x8, x0, [[REG]]
; CHECK: cmp x0, #0
; CHECK: csel x8, x8, x0, lt
; CHECK: asr x0, x8, #48
  %div = sdiv i64 %x, 281474976710656
  ret i64 %div
}

