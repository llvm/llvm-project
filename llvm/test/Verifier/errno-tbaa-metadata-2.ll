; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: assembly parsed, but does not verify as correct!
; CHECK-NEXT: Malformed struct tag metadata: base and access-type should be non-null and point to Metadata nodes
!llvm.errno.tbaa = !{!0}
!0 = !{!1, i64 0, !1}
!1 = !{!"int", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
