; RUN: opt < %s -passes=instcombine -S | FileCheck %s
;
; Test that instcombine folds allocsize function calls properly.
; Dummy arguments are inserted to verify that allocsize is picking the right
; args, and to prove that arbitrary unfoldable values don't interfere with
; allocsize if they're not used by allocsize.

declare ptr @my_malloc(ptr, i32) allocsize(1)
declare ptr @my_calloc(ptr, ptr, i32, i32) allocsize(2, 3)

; CHECK-LABEL: define void @test_malloc
define void @test_malloc(ptr %p, ptr %r) {
  %1 = call ptr @my_malloc(ptr null, i32 100)
  store ptr %1, ptr %p, align 8 ; To ensure objectsize isn't killed

  %2 = call i64 @llvm.objectsize.i64.p0(ptr %1, i1 false)
  ; CHECK: store i64 100
  store i64 %2, ptr %r, align 8
  ret void
}

; CHECK-LABEL: define void @test_calloc
define void @test_calloc(ptr %p, ptr %r) {
  %1 = call ptr @my_calloc(ptr null, ptr null, i32 100, i32 5)
  store ptr %1, ptr %p, align 8 ; To ensure objectsize isn't killed

  %2 = call i64 @llvm.objectsize.i64.p0(ptr %1, i1 false)
  ; CHECK: store i64 500
  store i64 %2, ptr %r, align 8
  ret void
}

; Failure cases with non-constant values...
; CHECK-LABEL: define void @test_malloc_fails
define void @test_malloc_fails(ptr %p, ptr %r, i32 %n) {
  %1 = call ptr @my_malloc(ptr null, i32 %n)
  store ptr %1, ptr %p, align 8 ; To ensure objectsize isn't killed

  ; CHECK: @llvm.objectsize.i64.p0
  %2 = call i64 @llvm.objectsize.i64.p0(ptr %1, i1 false)
  store i64 %2, ptr %r, align 8
  ret void
}

; CHECK-LABEL: define void @test_calloc_fails
define void @test_calloc_fails(ptr %p, ptr %r, i32 %n) {
  %1 = call ptr @my_calloc(ptr null, ptr null, i32 %n, i32 5)
  store ptr %1, ptr %p, align 8 ; To ensure objectsize isn't killed

  ; CHECK: @llvm.objectsize.i64.p0
  %2 = call i64 @llvm.objectsize.i64.p0(ptr %1, i1 false)
  store i64 %2, ptr %r, align 8


  %3 = call ptr @my_calloc(ptr null, ptr null, i32 100, i32 %n)
  store ptr %3, ptr %p, align 8 ; To ensure objectsize isn't killed

  ; CHECK: @llvm.objectsize.i64.p0
  %4 = call i64 @llvm.objectsize.i64.p0(ptr %3, i1 false)
  store i64 %4, ptr %r, align 8
  ret void
}

declare ptr @my_malloc_outofline(ptr, i32) #0
declare ptr @my_calloc_outofline(ptr, ptr, i32, i32) #1

; Verifying that out of line allocsize is parsed correctly
; CHECK-LABEL: define void @test_outofline
define void @test_outofline(ptr %p, ptr %r) {
  %1 = call ptr @my_malloc_outofline(ptr null, i32 100)
  store ptr %1, ptr %p, align 8 ; To ensure objectsize isn't killed

  %2 = call i64 @llvm.objectsize.i64.p0(ptr %1, i1 false)
  ; CHECK: store i64 100
  store i64 %2, ptr %r, align 8


  %3 = call ptr @my_calloc_outofline(ptr null, ptr null, i32 100, i32 5)
  store ptr %3, ptr %p, align 8 ; To ensure objectsize isn't killed

  %4 = call i64 @llvm.objectsize.i64.p0(ptr %3, i1 false)
  ; CHECK: store i64 500
  store i64 %4, ptr %r, align 8
  ret void
}

declare ptr @my_malloc_i64(ptr, i64) #0
declare ptr @my_tiny_calloc(ptr, ptr, i8, i8) #1
declare ptr @my_varied_calloc(ptr, ptr, i32, i8) #1

; CHECK-LABEL: define void @test_overflow
define void @test_overflow(ptr %p, ptr %r) {

  ; (2**31 + 1) * 2 > 2**31. So overflow. Yay.
  %big_malloc = call ptr @my_calloc(ptr null, ptr null, i32 2147483649, i32 2)
  store ptr %big_malloc, ptr %p, align 8

  ; CHECK: @llvm.objectsize
  %1 = call i32 @llvm.objectsize.i32.p0(ptr %big_malloc, i1 false)
  store i32 %1, ptr %r, align 4


  %big_little_malloc = call ptr @my_tiny_calloc(ptr null, ptr null, i8 127, i8 4)
  store ptr %big_little_malloc, ptr %p, align 8

  ; CHECK: store i32 508
  %2 = call i32 @llvm.objectsize.i32.p0(ptr %big_little_malloc, i1 false)
  store i32 %2, ptr %r, align 4


  ; malloc(2**33)
  %big_malloc_i64 = call ptr @my_malloc_i64(ptr null, i64 8589934592)
  store ptr %big_malloc_i64, ptr %p, align 8

  ; CHECK: @llvm.objectsize
  %3 = call i32 @llvm.objectsize.i32.p0(ptr %big_malloc_i64, i1 false)
  store i32 %3, ptr %r, align 4


  %4 = call i64 @llvm.objectsize.i64.p0(ptr %big_malloc_i64, i1 false)
  ; CHECK: store i64 8589934592
  store i64 %4, ptr %r, align 8


  ; Just intended to ensure that we properly handle args of different types...
  %varied_calloc = call ptr @my_varied_calloc(ptr null, ptr null, i32 1000, i8 5)
  store ptr %varied_calloc, ptr %p, align 8

  ; CHECK: store i32 5000
  %5 = call i32 @llvm.objectsize.i32.p0(ptr %varied_calloc, i1 false)
  store i32 %5, ptr %r, align 4

  ret void
}

; CHECK-LABEL: define void @test_nobuiltin
; We had a bug where `nobuiltin` would cause `allocsize` to be ignored in
; @llvm.objectsize calculations.
define void @test_nobuiltin(ptr %p, ptr %r) {
  %1 = call ptr @my_malloc(ptr null, i32 100) nobuiltin
  store ptr %1, ptr %p, align 8

  %2 = call i64 @llvm.objectsize.i64.p0(ptr %1, i1 false)
  ; CHECK: store i64 100
  store i64 %2, ptr %r, align 8
  ret void
}

attributes #0 = { allocsize(1) }
attributes #1 = { allocsize(2, 3) }

declare i32 @llvm.objectsize.i32.p0(ptr, i1)
declare i64 @llvm.objectsize.i64.p0(ptr, i1)
