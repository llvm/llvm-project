; RUN: llvm-as -disable-output < %s 2>&1 | FileCheck %s

!llvm.module.flags = !{!0, !1}

!llvm.dbg.retainedNodes = !{!2}

!0 = !{i32 7, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 4}

; CHECK: invalid module retained node
!2 = distinct !DIFragment()

; CHECK: warning: ignoring invalid debug info in <stdin>
