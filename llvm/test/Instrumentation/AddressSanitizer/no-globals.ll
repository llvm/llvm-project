; A module with no asan-instrumented globals has no asan destructor, and has an asan constructor in a comdat.
; RUN: opt -mtriple=x86_64-unknown-linux-gnu < %s -asan -asan-module -enable-new-pm=0 -asan-with-comdat=1 -asan-globals-live-support=1 -S | FileCheck %s
; RUN: opt -mtriple=x86_64-unknown-linux-gnu < %s -passes='asan-pipeline' -asan-with-comdat=1 -asan-globals-live-support=1 -S | FileCheck %s

define void @f() {
  ret void
}

; CHECK-NOT: @llvm.global_dtors
; CHECK: @llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 1, void ()* @asan.module_ctor, i8* bitcast (void ()* @asan.module_ctor to i8*) }]
; CHECK-NOT: @llvm.global_dtors
; CHECK: define internal void @asan.module_ctor() comdat
; CHECK-NOT: @llvm.global_dtors
