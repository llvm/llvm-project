; RUN: not llvm-link %s %p/module-flags-4-b.ll -S -o - 2>&1 | FileCheck %s

; Test 'require' error.

; CHECK: linking module flags 'bar': does not have the required value

!0 = metadata !{ i32 1, metadata !"foo", i32 37 }
!1 = metadata !{ i32 1, metadata !"bar", i32 927 }

!llvm.module.flags = !{ !0, !1 }
