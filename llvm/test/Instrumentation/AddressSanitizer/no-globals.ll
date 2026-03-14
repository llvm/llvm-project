; A module with no asan-instrumented globals has no asan destructor, and has an asan constructor in a comdat.
; RUN: opt -mtriple=x86_64-unknown-linux-gnu < %s -passes=asan -asan-with-comdat=1 -asan-globals-live-support=1 -S | FileCheck %s

define void @f() {
  ret void
}

; CHECK-NOT: @llvm.global_dtors
; CHECK: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1, ptr @asan.module_ctor, ptr @asan.module_ctor }]
; CHECK-NOT: @llvm.global_dtors
; CHECK: define internal void @asan.module_ctor() #[[#]] comdat
; CHECK-NOT: @llvm.global_dtors
