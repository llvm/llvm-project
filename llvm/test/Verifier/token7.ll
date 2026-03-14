; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

define void @f() {
entry:
  call token () undef ()
  ret void
}
; CHECK: Return type cannot be token for indirect call!
