; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

define void @test(ptr %p) {
  load i8, ptr %p, !noalias !0
  load i8, ptr %p, !noalias !1
  load i8, ptr %p, !noalias !3
  load i8, ptr %p, !noalias !5
  load i8, ptr %p, !noalias !7
  load i8, ptr %p, !noalias !9
  load i8, ptr %p, !noalias !11
  load i8, ptr %p, !noalias !14
  load i8, ptr %p, !noalias !17
  load i8, ptr %p, !noalias !20
  load i8, ptr %p, !noalias !23
  load i8, ptr %p, !alias.scope !26
  call void @llvm.experimental.noalias.scope.decl(metadata !29)
  ret void
}

declare void @llvm.experimental.noalias.scope.decl(metadata)

; CHECK: scope list must consist of MDNodes
!0 = !{!"str"}

; CHECK: scope must have two or three operands
!1 = !{!2}
!2 = !{!2}

; CHECK: scope must have two or three operands
!3 = !{!4}
!4 = !{!4, !5, !6, !7}

; CHECK: first scope operand must be self-referential or string
!5 = !{!6}
!6 = !{!7, !8}

; CHECK: third scope operand must be string (if used)
!7 = !{!8}
!8 = !{!8, !9, !10}

; CHECK: second scope operand must be MDNode
!9 = !{!10}
!10 = !{!10, !"str"}

; CHECK: domain must have two or three operands
!11 = !{!12}
!12 = !{!12, !13}
!13 = !{}

; CHECK: domain must have two or three operands
!14 = !{!15}
!15 = !{!15, !16}
!16 = !{!16, i1 false, !"str", !"str"}

; CHECK: first domain operand must be self-referential or string
!17 = !{!18}
!18 = !{!18, !19}
!19 = !{!13, i1 false}

; CHECK: second domain operand must be an i1 constant
!20 = !{!21}
!21 = !{!21, !22}
!22 = !{!22, i32 1}

; CHECK: second domain operand must be an i1 constant
!23 = !{!24}
!24 = !{!24, !25}
!25 = !{!25, !"str", !"str"}

; CHECK: third domain operand must be string (if used)
!26 = !{!27}
!27 = !{!27, !28}
!28 = !{!28, i1 false, !13}

; CHECK: domain must have two or three operands
!29 = !{!30}
!30 = !{!30, !31}
!31 = !{!31, i1 false, !"str", !"str"}
