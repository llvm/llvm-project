; RUN: not llvm-as %s -disable-output 2>&1 | FileCheck %s

define dso_local void @f() {
entry:
  %a.addr = alloca <2 x x86_amx>, align 4
  ret void
}

; CHECK: invalid vector element type
