; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

!named = !{!0}
; CHECK: Subrange must contain count or upperBound
!0 = !DISubrange(lowerBound: 1, stride: 4)
