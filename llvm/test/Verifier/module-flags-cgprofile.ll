; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

declare void @b()
declare void @a()

!llvm.module.flags = !{!0}

!0 = !{i32 5, !"CG Profile", !1}
!1 = !{!2, !"", !3, !4, !5, !6, !7, !8}
!2 = !{void ()* @a, void ()* @b, i64 32}
!3 = !{void ()* @a, void ()* @b}
!4 = !{void ()* @a, void ()* @b, i64 32, i64 32}
!5 = !{!"a", void ()* @b, i64 32}
!6 = !{void ()* @a, !"b", i64 32}
!7 = !{void ()* @a, void ()* @b, !""}
!8 = !{void ()* @a, void ()* @b, null}

; CHECK: expected a MDNode triple
; CHECK: !""
; CHECK: expected a MDNode triple
; CHECK: !3 = !{void ()* @a, void ()* @b}
; CHECK: expected a MDNode triple
; CHECK: !4 = !{void ()* @a, void ()* @b, i64 32, i64 32}
; CHECK: expected a Function or null
; CHECK: !"a"
; CHECK: expected a Function or null
; CHECK: !"b"
; CHECK: expected an integer constant
; CHECK: !""
; CHECK: expected an integer constant
