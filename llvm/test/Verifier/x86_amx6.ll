; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

define x86_amx @f() {
entry:
  ret x86_amx undef
}
; CHECK: Function returns a x86_amx but isn't an intrinsic
