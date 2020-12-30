; RUN: llc -mtriple=aarch64-linux-gnu -verify-machineinstrs -o - %s | FileCheck %s

@var32 = dso_local global [3 x i32] zeroinitializer
@var64 = dso_local global [3 x i64] zeroinitializer
@var32_align64 = dso_local global [3 x i32] zeroinitializer, align 8
@alias = dso_local alias [3 x i32], [3 x i32]* @var32_align64

define dso_local i64 @test_align32() {
; CHECK-LABEL: test_align32:
  %addr = bitcast [3 x i32]* @var32 to i64*

  ; Since @var32 is only guaranteed to be aligned to 32-bits, it's invalid to
  ; emit an "LDR x0, [x0, #:lo12:var32] instruction to implement this load.
  %val = load i64, i64* %addr
; CHECK: adrp [[HIBITS:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[HIBITS]], {{#?}}:lo12:var32
; CHECK: ldr x0, [x[[ADDR]]]

  ret i64 %val
}

define dso_local i64 @test_align64() {
; CHECK-LABEL: test_align64:
  %addr = bitcast [3 x i64]* @var64 to i64*

  ; However, var64 *is* properly aligned and emitting an adrp/add/ldr would be
  ; inefficient.
  %val = load i64, i64* %addr
; CHECK: adrp x[[HIBITS:[0-9]+]], var64
; CHECK-NOT: add x[[HIBITS]]
; CHECK: ldr x0, [x[[HIBITS]], {{#?}}:lo12:var64]

  ret i64 %val
}

define dso_local i64 @test_var32_align64() {
; CHECK-LABEL: test_var32_align64:
  %addr = bitcast [3 x i32]* @var32_align64 to i64*

  ; Since @var32 is only guaranteed to be aligned to 32-bits, it's invalid to
  ; emit an "LDR x0, [x0, #:lo12:var32] instruction to implement this load.
  %val = load i64, i64* %addr
; CHECK: adrp x[[HIBITS:[0-9]+]], var32_align64
; CHECK-NOT: add x[[HIBITS]]
; CHECK: ldr x0, [x[[HIBITS]], {{#?}}:lo12:var32_align64]

  ret i64 %val
}

define dso_local i64 @test_var32_alias() {
; CHECK-LABEL: test_var32_alias:
  %addr = bitcast [3 x i32]* @alias to i64*

  ; We don't know anything about the alignment of aliases.
  %val = load i64, i64* %addr
; CHECK: adrp x[[HIBITS:[0-9]+]], alias
; CHECK: add x[[ADDR:[0-9]+]], x[[HIBITS]], {{#?}}:lo12:alias
; CHECK: ldr x0, [x[[ADDR]]]

  ret i64 %val
}

@yet_another_var = external dso_local global {i32, i32}

define dso_local i64 @test_yet_another_var() {
; CHECK-LABEL: test_yet_another_var:

  ; @yet_another_var has a preferred alignment of 8, but that's not enough if
  ; we're going to be linking against other things. Its ABI alignment is only 4
  ; so we can't fold the load.
  %val = load i64, i64* bitcast({i32, i32}* @yet_another_var to i64*)
; CHECK: adrp [[HIBITS:x[0-9]+]], yet_another_var
; CHECK: add x[[ADDR:[0-9]+]], [[HIBITS]], {{#?}}:lo12:yet_another_var
; CHECK: ldr x0, [x[[ADDR]]]
  ret i64 %val
}

define dso_local i64()* @test_functions() {
; CHECK-LABEL: test_functions:
  ret i64()* @test_yet_another_var
; CHECK: adrp [[HIBITS:x[0-9]+]], test_yet_another_var
; CHECK: add x0, [[HIBITS]], {{#?}}:lo12:test_yet_another_var
}
