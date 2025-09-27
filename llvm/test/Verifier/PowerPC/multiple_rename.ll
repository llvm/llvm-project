; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

@a = global i32 1, section "abc", !rename !0, !rename !1

!0 = !{}
!1 = !{}

; CHECK: global value cannot have more then 1 rename metadata
