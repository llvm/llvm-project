; RUN: llvm-as -disable-output <%s 2>&1 | FileCheck %s
; CHECK: warning: ignoring invalid debug info

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !DIFile(filename: "davide.f", directory: "")
