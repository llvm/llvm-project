; RUN: llvm-as < %s -disable-output
; PR4654


@foo = constant i1 false
!0 = !{i1 false}
!a = !{!0}
