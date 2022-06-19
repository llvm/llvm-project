; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

define void @test(i8* %p) {
  load i8, i8* %p, !noalias !0
  load i8, i8* %p, !noalias !1
  load i8, i8* %p, !noalias !3
  load i8, i8* %p, !noalias !5
  load i8, i8* %p, !noalias !7
  load i8, i8* %p, !noalias !9
  load i8, i8* %p, !noalias !11
  load i8, i8* %p, !noalias !14
  load i8, i8* %p, !alias.scope !17
  call void @llvm.experimental.noalias.scope.decl(metadata !20)
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

; CHECK: domain must have one or two operands
!11 = !{!12}
!12 = !{!12, !13}
!13 = !{}

; CHECK: domain must have one or two operands
!14 = !{!15}
!15 = !{!15, !16}
!16 = !{!17, !18, !19}

; CHECK: first domain operand must be self-referential or string
!17 = !{!18}
!18 = !{!18, !19}
!19 = !{!20}

; CHECK: second domain operand must be string (if used)
!20 = !{!21}
!21 = !{!21, !22}
!22 = !{!22, !23}
!23 = !{}
