; RUN: llvm-link %s %p/Inputs/module-max-warn.ll -S -o - 2>&1 | FileCheck %s

; CHECK: warning: linking module flags 'Combine Max and Warn': IDs have conflicting values ('i32 4' from {{.*}}/Inputs/module-max-warn.ll with 'i32 2' from llvm-link)
; CHECK: warning: linking module flags 'Combine Warn and Max': IDs have conflicting values ('i32 5' from {{.*}}/Inputs/module-max-warn.ll with 'i32 3' from llvm-link)


; CHECK: !0 = !{i32 7, !"Combine Max and Warn", i32 4}
; CHECK: !1 = !{i32 7, !"Combine Warn and Max", i32 5}

!llvm.module.flags = !{!0, !1}
!0 = !{i32 7, !"Combine Max and Warn", i32 2}
!1 = !{i32 2, !"Combine Warn and Max", i32 3}
