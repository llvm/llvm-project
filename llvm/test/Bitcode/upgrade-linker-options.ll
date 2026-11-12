; RUN: llvm-as -disable-verify < %s | llvm-dis | FileCheck %s
; RUN: not llvm-as < %s 2>&1 | FileCheck --check-prefix=ERROR %s

; CHECK: !llvm.linker.options = !{!2, !3}
; CHECK: !2 = !{!"/DEFAULTLIB:libcmtd.lib"}
; CHECK: !3 = !{!"/DEFAULTLIB:oldnames.lib"}

; ERROR: 'Linker Options' named metadata no longer supported

!0 = !{i32 6, !"Linker Options", !1}
!1 = !{!2, !3}
!2 = !{!"/DEFAULTLIB:libcmtd.lib"}
!3 = !{!"/DEFAULTLIB:oldnames.lib"}

!llvm.module.flags = !{!0}
