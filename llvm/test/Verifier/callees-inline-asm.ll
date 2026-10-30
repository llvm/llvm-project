; RUN: not llvm-as -disable-output < %s 2>&1 | FileCheck %s

; CHECK: !callees metadata is not allowed on inline asm calls
define void @f() {
  call void asm sideeffect "", ""(), !callees !0
  ret void
}

declare void @g()

!0 = !{ptr @g}
