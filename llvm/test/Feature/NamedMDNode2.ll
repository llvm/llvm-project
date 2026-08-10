; RUN: llvm-as < %s -o /dev/null
; PR4654


@foo = constant i1 false
!0 = !{i1 false}
!a = !{!0}
