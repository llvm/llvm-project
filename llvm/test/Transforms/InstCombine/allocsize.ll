; RUN: opt < %s -passes=instcombine -S | FileCheck %s
;
; Test that instcombine folds allocsize function calls properly.
; Dummy arguments are inserted to verify that allocsize is picking the right
; args, and to prove that arbitrary unfoldable values don't interfere with
; allocsize if they're not used by allocsize.

declare i8* @my_malloc(i8*, i32) allocsize(1)
declare i8* @my_calloc(i8*, i8*, i32, i32) allocsize(2, 3)

; CHECK-LABEL: define void @test_malloc
define void @test_malloc(i8** %p, i64* %r) {
  %1 = call i8* @my_malloc(i8* null, i32 100)
  store i8* %1, i8** %p, align 8 ; To ensure objectsize isn't killed

  %2 = call i64 @llvm.objectsize.i64.p0i8(i8* %1, i1 false)
  ; CHECK: store i64 100
  store i64 %2, i64* %r, align 8
  ret void
}

; CHECK-LABEL: define void @test_calloc
define void @test_calloc(i8** %p, i64* %r) {
  %1 = call i8* @my_calloc(i8* null, i8* null, i32 100, i32 5)
  store i8* %1, i8** %p, align 8 ; To ensure objectsize isn't killed

  %2 = call i64 @llvm.objectsize.i64.p0i8(i8* %1, i1 false)
  ; CHECK: store i64 500
  store i64 %2, i64* %r, align 8
  ret void
}

; Failure cases with non-constant values...
; CHECK-LABEL: define void @test_malloc_fails
define void @test_malloc_fails(i8** %p, i64* %r, i32 %n) {
  %1 = call i8* @my_malloc(i8* null, i32 %n)
  store i8* %1, i8** %p, align 8 ; To ensure objectsize isn't killed

  ; CHECK: @llvm.objectsize.i64.p0i8
  %2 = call i64 @llvm.objectsize.i64.p0i8(i8* %1, i1 false)
  store i64 %2, i64* %r, align 8
  ret void
}

; CHECK-LABEL: define void @test_calloc_fails
define void @test_calloc_fails(i8** %p, i64* %r, i32 %n) {
  %1 = call i8* @my_calloc(i8* null, i8* null, i32 %n, i32 5)
  store i8* %1, i8** %p, align 8 ; To ensure objectsize isn't killed

  ; CHECK: @llvm.objectsize.i64.p0i8
  %2 = call i64 @llvm.objectsize.i64.p0i8(i8* %1, i1 false)
  store i64 %2, i64* %r, align 8


  %3 = call i8* @my_calloc(i8* null, i8* null, i32 100, i32 %n)
  store i8* %3, i8** %p, align 8 ; To ensure objectsize isn't killed

  ; CHECK: @llvm.objectsize.i64.p0i8
  %4 = call i64 @llvm.objectsize.i64.p0i8(i8* %3, i1 false)
  store i64 %4, i64* %r, align 8
  ret void
}

declare i8* @my_malloc_outofline(i8*, i32) #0
declare i8* @my_calloc_outofline(i8*, i8*, i32, i32) #1

; Verifying that out of line allocsize is parsed correctly
; CHECK-LABEL: define void @test_outofline
define void @test_outofline(i8** %p, i64* %r) {
  %1 = call i8* @my_malloc_outofline(i8* null, i32 100)
  store i8* %1, i8** %p, align 8 ; To ensure objectsize isn't killed

  %2 = call i64 @llvm.objectsize.i64.p0i8(i8* %1, i1 false)
  ; CHECK: store i64 100
  store i64 %2, i64* %r, align 8


  %3 = call i8* @my_calloc_outofline(i8* null, i8* null, i32 100, i32 5)
  store i8* %3, i8** %p, align 8 ; To ensure objectsize isn't killed

  %4 = call i64 @llvm.objectsize.i64.p0i8(i8* %3, i1 false)
  ; CHECK: store i64 500
  store i64 %4, i64* %r, align 8
  ret void
}

declare i8* @my_malloc_i64(i8*, i64) #0
declare i8* @my_tiny_calloc(i8*, i8*, i8, i8) #1
declare i8* @my_varied_calloc(i8*, i8*, i32, i8) #1

; CHECK-LABEL: define void @test_overflow
define void @test_overflow(i8** %p, i32* %r) {
  %r64 = bitcast i32* %r to i64*

  ; (2**31 + 1) * 2 > 2**31. So overflow. Yay.
  %big_malloc = call i8* @my_calloc(i8* null, i8* null, i32 2147483649, i32 2)
  store i8* %big_malloc, i8** %p, align 8

  ; CHECK: @llvm.objectsize
  %1 = call i32 @llvm.objectsize.i32.p0i8(i8* %big_malloc, i1 false)
  store i32 %1, i32* %r, align 4


  %big_little_malloc = call i8* @my_tiny_calloc(i8* null, i8* null, i8 127, i8 4)
  store i8* %big_little_malloc, i8** %p, align 8

  ; CHECK: store i32 508
  %2 = call i32 @llvm.objectsize.i32.p0i8(i8* %big_little_malloc, i1 false)
  store i32 %2, i32* %r, align 4


  ; malloc(2**33)
  %big_malloc_i64 = call i8* @my_malloc_i64(i8* null, i64 8589934592)
  store i8* %big_malloc_i64, i8** %p, align 8

  ; CHECK: @llvm.objectsize
  %3 = call i32 @llvm.objectsize.i32.p0i8(i8* %big_malloc_i64, i1 false)
  store i32 %3, i32* %r, align 4


  %4 = call i64 @llvm.objectsize.i64.p0i8(i8* %big_malloc_i64, i1 false)
  ; CHECK: store i64 8589934592
  store i64 %4, i64* %r64, align 8


  ; Just intended to ensure that we properly handle args of different types...
  %varied_calloc = call i8* @my_varied_calloc(i8* null, i8* null, i32 1000, i8 5)
  store i8* %varied_calloc, i8** %p, align 8

  ; CHECK: store i32 5000
  %5 = call i32 @llvm.objectsize.i32.p0i8(i8* %varied_calloc, i1 false)
  store i32 %5, i32* %r, align 4

  ret void
}

; CHECK-LABEL: define void @test_nobuiltin
; We had a bug where `nobuiltin` would cause `allocsize` to be ignored in
; @llvm.objectsize calculations.
define void @test_nobuiltin(i8** %p, i64* %r) {
  %1 = call i8* @my_malloc(i8* null, i32 100) nobuiltin
  store i8* %1, i8** %p, align 8

  %2 = call i64 @llvm.objectsize.i64.p0i8(i8* %1, i1 false)
  ; CHECK: store i64 100
  store i64 %2, i64* %r, align 8
  ret void
}

attributes #0 = { allocsize(1) }
attributes #1 = { allocsize(2, 3) }

declare i32 @llvm.objectsize.i32.p0i8(i8*, i1)
declare i64 @llvm.objectsize.i64.p0i8(i8*, i1)
