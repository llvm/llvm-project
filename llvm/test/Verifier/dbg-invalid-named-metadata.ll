; RUN: not llvm-as -disable-output <%s 2>&1 | FileCheck %s
; CHECK:      assembly parsed, but does not verify

!llvm.module.flags = !{!0}
!llvm.dbg.the_dbg_namespace_is_reserved = !{}

!0 = !{i32 2, !"Debug Info Version", i32 3}
