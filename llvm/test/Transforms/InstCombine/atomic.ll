; RUN: opt -S < %s -instcombine | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.7.0"

; Check transforms involving atomic operations

define i32 @test1(i32* %p) {
; CHECK-LABEL: define i32 @test1(
; CHECK: %x = load atomic i32, i32* %p seq_cst, align 4
; CHECK: shl i32 %x, 1
  %x = load atomic i32, i32* %p seq_cst, align 4
  %y = load i32, i32* %p, align 4
  %z = add i32 %x, %y
  ret i32 %z
}

define i32 @test2(i32* %p) {
; CHECK-LABEL: define i32 @test2(
; CHECK: %x = load volatile i32, i32* %p, align 4
; CHECK: %y = load volatile i32, i32* %p, align 4
  %x = load volatile i32, i32* %p, align 4
  %y = load volatile i32, i32* %p, align 4
  %z = add i32 %x, %y
  ret i32 %z
}

; The exact semantics of mixing volatile and non-volatile on the same
; memory location are a bit unclear, but conservatively, we know we don't
; want to remove the volatile.
define i32 @test3(i32* %p) {
; CHECK-LABEL: define i32 @test3(
; CHECK: %x = load volatile i32, i32* %p, align 4
  %x = load volatile i32, i32* %p, align 4
  %y = load i32, i32* %p, align 4
  %z = add i32 %x, %y
  ret i32 %z
}

; Forwarding from a stronger ordered atomic is fine
define i32 @test4(i32* %p) {
; CHECK-LABEL: define i32 @test4(
; CHECK: %x = load atomic i32, i32* %p seq_cst, align 4
; CHECK: shl i32 %x, 1
  %x = load atomic i32, i32* %p seq_cst, align 4
  %y = load atomic i32, i32* %p unordered, align 4
  %z = add i32 %x, %y
  ret i32 %z
}

; Forwarding from a non-atomic is not.  (The earlier load 
; could in priciple be promoted to atomic and then forwarded, 
; but we can't just  drop the atomic from the load.)
define i32 @test5(i32* %p) {
; CHECK-LABEL: define i32 @test5(
; CHECK: %x = load atomic i32, i32* %p unordered, align 4
  %x = load atomic i32, i32* %p unordered, align 4
  %y = load i32, i32* %p, align 4
  %z = add i32 %x, %y
  ret i32 %z
}

; Forwarding atomic to atomic is fine
define i32 @test6(i32* %p) {
; CHECK-LABEL: define i32 @test6(
; CHECK: %x = load atomic i32, i32* %p unordered, align 4
; CHECK: shl i32 %x, 1
  %x = load atomic i32, i32* %p unordered, align 4
  %y = load atomic i32, i32* %p unordered, align 4
  %z = add i32 %x, %y
  ret i32 %z
}

; FIXME: we currently don't do anything for monotonic
define i32 @test7(i32* %p) {
; CHECK-LABEL: define i32 @test7(
; CHECK: %x = load atomic i32, i32* %p seq_cst, align 4
; CHECK: %y = load atomic i32, i32* %p monotonic, align 4
  %x = load atomic i32, i32* %p seq_cst, align 4
  %y = load atomic i32, i32* %p monotonic, align 4
  %z = add i32 %x, %y
  ret i32 %z
}

; FIXME: We could forward in racy code
define i32 @test8(i32* %p) {
; CHECK-LABEL: define i32 @test8(
; CHECK: %x = load atomic i32, i32* %p seq_cst, align 4
; CHECK: %y = load atomic i32, i32* %p acquire, align 4
  %x = load atomic i32, i32* %p seq_cst, align 4
  %y = load atomic i32, i32* %p acquire, align 4
  %z = add i32 %x, %y
  ret i32 %z
}

; An unordered access to null is still unreachable.  There's no
; ordering imposed.
define i32 @test9() {
; CHECK-LABEL: define i32 @test9(
; CHECK: store i32 undef, i32* null
  %x = load atomic i32, i32* null unordered, align 4
  ret i32 %x
}

; FIXME: Could also fold
define i32 @test10() {
; CHECK-LABEL: define i32 @test10(
; CHECK: load atomic i32, i32* null monotonic
  %x = load atomic i32, i32* null monotonic, align 4
  ret i32 %x
}

; Would this be legal to fold?  Probably?
define i32 @test11() {
; CHECK-LABEL: define i32 @test11(
; CHECK: load atomic i32, i32* null seq_cst
  %x = load atomic i32, i32* null seq_cst, align 4
  ret i32 %x
}

; An unordered access to null is still unreachable.  There's no
; ordering imposed.
define i32 @test12() {
; CHECK-LABEL: define i32 @test12(
; CHECK: store atomic i32 undef, i32* null
  store atomic i32 0, i32* null unordered, align 4
  ret i32 0
}

; FIXME: Could also fold
define i32 @test13() {
; CHECK-LABEL: define i32 @test13(
; CHECK: store atomic i32 0, i32* null monotonic
  store atomic i32 0, i32* null monotonic, align 4
  ret i32 0
}

; Would this be legal to fold?  Probably?
define i32 @test14() {
; CHECK-LABEL: define i32 @test14(
; CHECK: store atomic i32 0, i32* null seq_cst
  store atomic i32 0, i32* null seq_cst, align 4
  ret i32 0
}

@a = external global i32
@b = external global i32

define i32 @test15(i1 %cnd) {
; CHECK-LABEL: define i32 @test15(
; CHECK: load atomic i32, i32* @a unordered, align 4
; CHECK: load atomic i32, i32* @b unordered, align 4
  %addr = select i1 %cnd, i32* @a, i32* @b
  %x = load atomic i32, i32* %addr unordered, align 4
  ret i32 %x
}

; FIXME: This would be legal to transform
define i32 @test16(i1 %cnd) {
; CHECK-LABEL: define i32 @test16(
; CHECK: load atomic i32, i32* %addr monotonic, align 4
  %addr = select i1 %cnd, i32* @a, i32* @b
  %x = load atomic i32, i32* %addr monotonic, align 4
  ret i32 %x
}

; FIXME: This would be legal to transform
define i32 @test17(i1 %cnd) {
; CHECK-LABEL: define i32 @test17(
; CHECK: load atomic i32, i32* %addr seq_cst, align 4
  %addr = select i1 %cnd, i32* @a, i32* @b
  %x = load atomic i32, i32* %addr seq_cst, align 4
  ret i32 %x
}
