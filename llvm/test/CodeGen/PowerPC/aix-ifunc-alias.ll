; XFAIL: *
; RUN: llc -mtriple=powerpc64-ibm-aix-xcoff %s -o - | FileCheck %s

@foo_alias = alias i32 (...), ptr @my_foo
@foo = ifunc i32 (...), ptr @foo.resolver

define hidden i32 @my_foo() {
entry:
  ret i32 4
}

define internal ptr @foo.resolver() {
entry:
  ret ptr @my_foo
}

