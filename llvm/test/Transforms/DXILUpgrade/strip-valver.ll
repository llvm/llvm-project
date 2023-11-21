; RUN: opt -passes=dxil-upgrade -S < %s | FileCheck %s

; Ensure that both the valver metadata and its operand are removed.
; CHECK: !unrelated_md1 = !{!0}
; CHECK-NOT: !dx.valver
; CHECK: !unrelated_md2 = !{!1}
;
; CHECK: !0 = !{i32 1234}
; CHECK-NOT: !{i32 1, i32 7}
; CHECK: !1 = !{i32 4321}

!unrelated_md1 = !{!0}
!dx.valver = !{!1}
!unrelated_md2 = !{!2}

!0 = !{i32 1234}
!1 = !{i32 1, i32 7}
!2 = !{i32 4321}
