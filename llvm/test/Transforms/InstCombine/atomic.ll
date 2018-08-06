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

define i32 @test9_no_null_opt() #0 {
; CHECK-LABEL: define i32 @test9_no_null_opt(
; CHECK: load atomic i32, i32* null unordered
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

define i32 @test10_no_null_opt() #0 {
; CHECK-LABEL: define i32 @test10_no_null_opt(
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

define i32 @test11_no_null_opt() #0 {
; CHECK-LABEL: define i32 @test11_no_null_opt(
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

define i32 @test12_no_null_opt() #0 {
; CHECK-LABEL: define i32 @test12_no_null_opt(
; CHECK: store atomic i32 0, i32* null unordered
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

define i32 @test13_no_null_opt() #0 {
; CHECK-LABEL: define i32 @test13_no_null_opt(
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

define i32 @test14_no_null_opt() #0 {
; CHECK-LABEL: define i32 @test14_no_null_opt(
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

define i32 @test22(i1 %cnd) {
; CHECK-LABEL: define i32 @test22(
; CHECK: [[PHI:%.*]] = phi i32
; CHECK: store atomic i32 [[PHI]], i32* @a unordered, align 4
  br i1 %cnd, label %block1, label %block2

block1:
  store atomic i32 1, i32* @a unordered, align 4
  br label %merge
block2:
  store atomic i32 2, i32* @a unordered, align 4
  br label %merge

merge:
  ret i32 0
}

; TODO: probably also legal here
define i32 @test23(i1 %cnd) {
; CHECK-LABEL: define i32 @test23(
; CHECK: br i1 %cnd, label %block1, label %block2
  br i1 %cnd, label %block1, label %block2

block1:
  store atomic i32 1, i32* @a monotonic, align 4
  br label %merge
block2:
  store atomic i32 2, i32* @a monotonic, align 4
  br label %merge

merge:
  ret i32 0
}

declare void @clobber()

define i32 @test18(float* %p) {
; CHECK-LABEL: define i32 @test18(
; CHECK: load atomic i32, i32* [[A:%.*]] unordered, align 4
; CHECK: store atomic i32 [[B:%.*]], i32* [[C:%.*]] unordered, align 4
  %x = load atomic float, float* %p unordered, align 4
  call void @clobber() ;; keep the load around
  store atomic float %x, float* %p unordered, align 4
  ret i32 0
}

; TODO: probably also legal in this case
define i32 @test19(float* %p) {
; CHECK-LABEL: define i32 @test19(
; CHECK: load atomic float, float* %p seq_cst, align 4
; CHECK: store atomic float %x, float* %p seq_cst, align 4
  %x = load atomic float, float* %p seq_cst, align 4
  call void @clobber() ;; keep the load around
  store atomic float %x, float* %p seq_cst, align 4
  ret i32 0
}

define i32 @test20(i32** %p, i8* %v) {
; CHECK-LABEL: define i32 @test20(
; CHECK: store atomic i8* %v, i8** [[D:%.*]] unordered, align 4
  %cast = bitcast i8* %v to i32*
  store atomic i32* %cast, i32** %p unordered, align 4
  ret i32 0
}

define i32 @test21(i32** %p, i8* %v) {
; CHECK-LABEL: define i32 @test21(
; CHECK: store atomic i32* %cast, i32** %p monotonic, align 4
  %cast = bitcast i8* %v to i32*
  store atomic i32* %cast, i32** %p monotonic, align 4
  ret i32 0
}

define void @pr27490a(i8** %p1, i8** %p2) {
; CHECK-LABEL: define void @pr27490
; CHECK: %1 = bitcast i8** %p1 to i64*
; CHECK: %l1 = load i64, i64* %1, align 8
; CHECK: %2 = bitcast i8** %p2 to i64*
; CHECK: store volatile i64 %l1, i64* %2, align 8
  %l = load i8*, i8** %p1
  store volatile i8* %l, i8** %p2
  ret void
}

define void @pr27490b(i8** %p1, i8** %p2) {
; CHECK-LABEL: define void @pr27490
; CHECK: %1 = bitcast i8** %p1 to i64*
; CHECK: %l1 = load i64, i64* %1, align 8
; CHECK: %2 = bitcast i8** %p2 to i64*
; CHECK: store atomic i64 %l1, i64* %2 seq_cst, align 8
  %l = load i8*, i8** %p1
  store atomic i8* %l, i8** %p2 seq_cst, align 8
  ret void
}

;; At the moment, we can't form atomic vectors by folding since these are 
;; not representable in the IR.  This was pr29121.  The right long term
;; solution is to extend the IR to handle this case.
define <2 x float> @no_atomic_vector_load(i64* %p) {
; CHECK-LABEL @no_atomic_vector_load
; CHECK: load atomic i64, i64* %p unordered, align 8
  %load = load atomic i64, i64* %p unordered, align 8
  %.cast = bitcast i64 %load to <2 x float>
  ret <2 x float> %.cast
}

define void @no_atomic_vector_store(<2 x float> %p, i8* %p2) {
; CHECK-LABEL: @no_atomic_vector_store
; CHECK: store atomic i64 %1, i64* %2 unordered, align 8
  %1 = bitcast <2 x float> %p to i64
  %2 = bitcast i8* %p2 to i64*
  store atomic i64 %1, i64* %2 unordered, align 8
  ret void
}

attributes #0 = { "null-pointer-is-valid"="true" }
