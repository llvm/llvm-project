; RUN: opt -S -mtriple=aarch64-pc-windows-msvc -passes='default<O0>' < %s | FileCheck %s --check-prefix=COFF
; RUN: opt -S -mtriple=aarch64-pc-windows-msvc -passes='default<O2>' < %s | FileCheck %s --check-prefix=COFF
; RUN: opt -S -mtriple=aarch64-unknown-uefi -passes='default<O0>' < %s | FileCheck %s --check-prefix=COFF
; RUN: opt -S -mtriple=aarch64-unknown-uefi -passes='default<O2>' < %s | FileCheck %s --check-prefix=COFF
; RUN: opt -S -mtriple=aarch64-unknown-linux-gnu -passes='default<O2>' < %s | FileCheck %s --check-prefix=ELF

; COFF has no ifunc support, so the AArch64 target lowers ifuncs to a function
; pointer that a global constructor initialises from the resolver on both
; Windows and UEFI. ELF targets keep the ifunc.

; COFF-NOT: ifunc
; COFF: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 10, ptr @[[CTOR:.*]], ptr null }]
; COFF-LABEL: define {{.*}}i32 @bar()
; COFF: [[FN:%.*]] = load ptr, ptr @
; COFF: call i32 [[FN]](i32 1)
; COFF: define internal void @[[CTOR]]()
; COFF-NOT: ifunc

; ELF: @foo = ifunc i32 (i32), ptr @foo_resolver
; ELF-LABEL: define {{.*}}i32 @bar()
; ELF: call i32 @foo(i32 1)

@global = external global i32

@foo = ifunc i32 (i32), ptr @foo_resolver

define internal ptr @foo_resolver() {
  %v = load i32, ptr @global
  %c = icmp ne i32 %v, 0
  %r = select i1 %c, ptr @f1, ptr @f2
  ret ptr %r
}

define internal i32 @f1(i32 %i) {
  %a = add i32 %i, 1
  ret i32 %a
}

define internal i32 @f2(i32 %i) {
  %a = add i32 %i, 2
  ret i32 %a
}

define i32 @bar() {
  %call = call i32 @foo(i32 1)
  ret i32 %call
}
