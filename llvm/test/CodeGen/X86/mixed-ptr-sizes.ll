; RUN: llc < %s | FileCheck --check-prefixes=CHECK %s
; RUN: llc -O0 < %s | FileCheck --check-prefixes=CHECK %s

; Source to regenerate:
; struct Foo {
;   int * __ptr32 p32;
;   int * __ptr64 p64;
;   __attribute__((address_space(9))) int *p_other;
; };
; void use_foo(Foo *f);
; void test_sign_ext(Foo *f, int * __ptr32 __sptr i) {
;   f->p64 = i;
;   use_foo(f);
; }
; void test_zero_ext(Foo *f, int * __ptr32 __uptr i) {
;   f->p64 = i;
;   use_foo(f);
; }
; void test_trunc(Foo *f, int * __ptr64 i) {
;   f->p32 = i;
;   use_foo(f);
; }
; void test_noop1(Foo *f, int * __ptr32 i) {
;   f->p32 = i;
;   use_foo(f);
; }
; void test_noop2(Foo *f, int * __ptr64 i) {
;   f->p64 = i;
;   use_foo(f);
; }
; void test_null_arg(Foo *f, int * __ptr32 i) {
;   test_noop1(f, 0);
; }
; void test_unrecognized(Foo *f, __attribute__((address_space(14))) int *i) {
;   f->p32 = (int * __ptr32)i;
;   use_foo(f);
; }
;
; $ clang -cc1 -triple x86_64-windows-msvc -fms-extensions -O2 -S t.cpp

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-windows-msvc"

%struct.Foo = type { i32 addrspace(270)*, i32*, i32 addrspace(9)* }
declare dso_local void @use_foo(%struct.Foo*)

define dso_local void @test_sign_ext(%struct.Foo* %f, i32 addrspace(270)* %i) {
; CHECK-LABEL: test_sign_ext
; CHECK:       movslq %edx, %rax
entry:
  %0 = addrspacecast i32 addrspace(270)* %i to i32*
  %p64 = getelementptr inbounds %struct.Foo, %struct.Foo* %f, i64 0, i32 1
  store i32* %0, i32** %p64, align 8
  tail call void @use_foo(%struct.Foo* %f)
  ret void
}

define dso_local void @test_zero_ext(%struct.Foo* %f, i32 addrspace(271)* %i) {
; CHECK-LABEL: test_zero_ext
; CHECK:       movl %edx, %eax
entry:
  %0 = addrspacecast i32 addrspace(271)* %i to i32*
  %p64 = getelementptr inbounds %struct.Foo, %struct.Foo* %f, i64 0, i32 1
  store i32* %0, i32** %p64, align 8
  tail call void @use_foo(%struct.Foo* %f)
  ret void
}

define dso_local void @test_trunc(%struct.Foo* %f, i32* %i) {
; CHECK-LABEL: test_trunc
; CHECK:       movl %edx, (%rcx)
entry:
  %0 = addrspacecast i32* %i to i32 addrspace(270)*
  %p32 = getelementptr inbounds %struct.Foo, %struct.Foo* %f, i64 0, i32 0
  store i32 addrspace(270)* %0, i32 addrspace(270)** %p32, align 8
  tail call void @use_foo(%struct.Foo* %f)
  ret void
}

define dso_local void @test_noop1(%struct.Foo* %f, i32 addrspace(270)* %i) {
; CHECK-LABEL: test_noop1
; CHECK:       movl %edx, (%rcx)
entry:
  %p32 = getelementptr inbounds %struct.Foo, %struct.Foo* %f, i64 0, i32 0
  store i32 addrspace(270)* %i, i32 addrspace(270)** %p32, align 8
  tail call void @use_foo(%struct.Foo* %f)
  ret void
}

define dso_local void @test_noop2(%struct.Foo* %f, i32* %i) {
; CHECK-LABEL: test_noop2
; CHECK:       movq %rdx, 8(%rcx)
entry:
  %p64 = getelementptr inbounds %struct.Foo, %struct.Foo* %f, i64 0, i32 1
  store i32* %i, i32** %p64, align 8
  tail call void @use_foo(%struct.Foo* %f)
  ret void
}

; Test that null can be passed as a 32-bit pointer.
define dso_local void @test_null_arg(%struct.Foo* %f) {
entry:
  call void @test_noop1(%struct.Foo* %f, i32 addrspace(270)* null)
  ret void
}

; Test casts between unrecognized address spaces.
define void @test_unrecognized(%struct.Foo* %f, i32 addrspace(14)* %i) {
; CHECK-LABEL: test_unrecognized
; CHECK:       movl %edx, (%rcx)
entry:
  %0 = addrspacecast i32 addrspace(14)* %i to i32 addrspace(270)*
  %p32 = getelementptr inbounds %struct.Foo, %struct.Foo* %f, i64 0, i32 0
  store i32 addrspace(270)* %0, i32 addrspace(270)** %p32, align 8
  tail call void @use_foo(%struct.Foo* %f)
  ret void
}

define void @test_unrecognized2(%struct.Foo* %f, i32 addrspace(271)* %i) {
; CHECK-LABEL: test_unrecognized2
; CHECK:       movl %edx, %eax
entry:
  %0 = addrspacecast i32 addrspace(271)* %i to i32 addrspace(9)*
  %p32 = getelementptr inbounds %struct.Foo, %struct.Foo* %f, i64 0, i32 2
  store i32 addrspace(9)* %0, i32 addrspace(9)** %p32, align 8
  tail call void @use_foo(%struct.Foo* %f)
  ret void
}
