; RUN:  llvm-dis < %s.bc| FileCheck %s

; Apple Internal: Upgrade files with no ptrauth-abi-version to version -1.
; CHECK: !llvm.module.flags = !{!0}
; CHECK: !0 = !{i32 6, !"ptrauth.abi-version", !1}
; CHECK: !1 = !{!2}
; CHECK: !2 = !{i32 -1, i1 false}
