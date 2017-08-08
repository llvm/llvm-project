; RUN: not llvm-as -disable-output <%s 2>&1 | FileCheck %s
; CHECK:      assembly parsed, but does not verify

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !DIFile(filename: "davide.f", directory: "")
