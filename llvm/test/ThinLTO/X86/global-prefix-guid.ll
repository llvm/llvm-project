; REQUIRES: x86-registered-target

; The linker name of foo is _foo here, which is also the IR name of another
; function. foo must not get the GUID of that function and be dropped.

; RUN: opt -module-summary %s -o %t.bc
; RUN: llvm-lto2 run %t.bc -o %t.out -r=%t.bc,_foo,px -r=%t.bc,__foo,px
; RUN: llvm-nm %t.out.1 | FileCheck %s

; CHECK-DAG: T __foo
; CHECK-DAG: T _foo

target datalayout = "e-m:x-p:32:32-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc"

define dso_local i32 @foo(i32 %x) {
  %r = or i32 %x, 32
  ret i32 %r
}

define dso_local i32 @_foo(i32 %x) {
  %r = or i32 %x, 33
  ret i32 %r
}
