; RUN: llvm-as %s -o /dev/null

; Ensure context-owned constants are destroyed before their metadata wrappers.

!named = !{!0}
!0 = !{token none}
