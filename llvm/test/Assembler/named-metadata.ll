; RUN: llvm-as < %s | llvm-dis | FileCheck %s

!0 = metadata !{metadata !"zero"}
!1 = metadata !{metadata !"one"}
!2 = metadata !{metadata !"two"}

!foo = !{!0, !1, !2}
; CHECK: !foo = !{!0, !1, !2}

!\23pragma = !{!0, !1, !2}
; CHECK: !\23pragma = !{!0, !1, !2}

; \31 is the digit '1'. On emission, we escape the first character (to avoid
; conflicting with anonymous metadata), but not the subsequent ones.
!\31\31\31 = !{!0, !1, !2}
; CHECK: !\3111 = !{!0, !1, !2}

!\22name\22 = !{!0, !1, !2}
; CHECK: !\22name\22 = !{!0, !1, !2}

; \x doesn't mean anything, so we parse it literally but escape the \ into \5C
; when emitting it, followed by xfoo.
!\xfoo = !{!0, !1, !2}
; CHECK: !\5Cxfoo = !{!0, !1, !2}
