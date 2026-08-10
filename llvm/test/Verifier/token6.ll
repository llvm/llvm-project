; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

define token @f() {
entry:
  ret token undef
}
; CHECK: Function returns a token but isn't an intrinsic
