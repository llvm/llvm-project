; Test aliasing errors on z/OS

; RUN: not llc < %s -mtriple=s390x-ibm-zos 2>&1 | FileCheck %s

; CHECK: error: Only aliases to functions is supported in GOFF.
; CHECK: error: Weak alias/reference not supported on z/OS

@actual_variable = global i32 0
@alias_variable = alias i32, ptr @actual_variable

@foo1 = weak alias i32 (i32), ptr @foo
define hidden void @foo() {
entry:
  ret void
}

