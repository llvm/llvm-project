; RUN: not llvm-link %s %p/module-flags-4-b.ll -S -o - 2>&1 | FileCheck %s

; Test 'require' error.

; CHECK: linking module flags 'bar': does not have the required value

!0 = !{ i32 1, !"foo", i32 37 }
!1 = !{ i32 4, !"bar", i32 927 }

!llvm.module.flags = !{ !0, !1 }
