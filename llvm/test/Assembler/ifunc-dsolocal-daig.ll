; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

@foo = dso_local ifunc i32 (i32), i64 ()* @foo_ifunc
; CHECK: error: dso_local is invalid on ifunc

define internal i64 @foo_ifunc() {
entry:
  ret i64 0
}
