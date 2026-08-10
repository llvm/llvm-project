; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

define dso_local void @f() {
entry:
  %a.addr = alloca <2 x x86_amx>, align 4
  ret void
}

; CHECK: invalid vector element type
