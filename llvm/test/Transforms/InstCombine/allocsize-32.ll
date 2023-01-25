; RUN: opt < %s -passes=instcombine -S | FileCheck %s
;
; The idea is that we want to have sane semantics (e.g. not assertion failures)
; when given an allocsize function that takes a 64-bit argument in the face of
; 32-bit pointers.

target datalayout="e-p:32:32:32"

declare ptr @my_malloc(ptr, i64) allocsize(1)

define void @test_malloc(ptr %p, ptr %r) {
  %1 = call ptr @my_malloc(ptr null, i64 100)
  store ptr %1, ptr %p, align 8 ; To ensure objectsize isn't killed

  %2 = call i32 @llvm.objectsize.i32.p0(ptr %1, i1 false)
  ; CHECK: store i32 100
  store i32 %2, ptr %r, align 8

  ; Big number is 5 billion.
  %3 = call ptr @my_malloc(ptr null, i64 5000000000)
  store ptr %3, ptr %p, align 8 ; To ensure objectsize isn't killed

  ; CHECK: call i32 @llvm.objectsize
  %4 = call i32 @llvm.objectsize.i32.p0(ptr %3, i1 false)
  store i32 %4, ptr %r, align 8
  ret void
}

declare i32 @llvm.objectsize.i32.p0(ptr, i1)
