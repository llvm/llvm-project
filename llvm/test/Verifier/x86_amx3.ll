; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

define void @f(x86_amx %A, x86_amx %B) {
entry:
  alloca x86_amx
; CHECK: invalid type for alloca
  ret void
}
