; RUN: opt < %s -S | FileCheck %s

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"Debug Info Version", !""}
; CHECK: !{i32 1, !"Debug Info Version", !""}
