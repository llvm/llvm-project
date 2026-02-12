; RUN: llvm-as < %s > %t
; RUN: llvm-lto -list-symbols-only %t | FileCheck %s

; CHECK: foo    { function defined default }

target triple = "powerpc-ibm-aix7.2.0.0"

@foo = ifunc i32 (...), ptr @foo.resolver

define internal ptr @foo.resolver() {
entry:
  ret ptr @my_foo2
}

define internal i32 @my_foo2() {
entry:
  ret i32 5
}
