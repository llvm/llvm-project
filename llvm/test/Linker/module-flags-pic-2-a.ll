; RUN: llvm-link %s %p/Inputs/module-flags-pic-2-b.ll -S -o - | FileCheck %s

; test linking modules with two different PIC and PIE levels

!0 = !{ i32 8, !"PIC Level", i32 1 }
!1 = !{ i32 7, !"PIE Level", i32 1 }

!llvm.module.flags = !{!0, !1}

; CHECK: !0 = !{i32 8, !"PIC Level", i32 1}
; CHECK: !1 = !{i32 7, !"PIE Level", i32 2}
