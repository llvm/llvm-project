; RUN: not llvm-as %s -disable-output 2>&1 | FileCheck %s

define void @f() {
entry:
  call x86_amx () undef ()
  ret void
}
; CHECK: Return type cannot be x86_amx for indirect call!
