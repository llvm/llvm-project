; RUN: opt < %s -passes=instcombine -S | FileCheck %s
;
; The idea is that we want to have sane semantics (e.g. not assertion failures)
; when given an allocsize function that takes a 64-bit argument in the face of
; 32-bit pointers.

target datalayout="e-p:32:32:32"

declare i8* @my_malloc(i8*, i64) allocsize(1)

define void @test_malloc(i8** %p, i32* %r) {
  %1 = call i8* @my_malloc(i8* null, i64 100)
  store i8* %1, i8** %p, align 8 ; To ensure objectsize isn't killed

  %2 = call i32 @llvm.objectsize.i32.p0i8(i8* %1, i1 false)
  ; CHECK: store i32 100
  store i32 %2, i32* %r, align 8

  ; Big number is 5 billion.
  %3 = call i8* @my_malloc(i8* null, i64 5000000000)
  store i8* %3, i8** %p, align 8 ; To ensure objectsize isn't killed

  ; CHECK: call i32 @llvm.objectsize
  %4 = call i32 @llvm.objectsize.i32.p0i8(i8* %3, i1 false)
  store i32 %4, i32* %r, align 8
  ret void
}

declare i32 @llvm.objectsize.i32.p0i8(i8*, i1)
