; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder < %s

!llvm.module.flags = !{!0}

; CHECK: !0 = !{i32 8, !"PIC Level", i32 1}

!0 = !{i32 7, !"PIC Level", i32 1}
