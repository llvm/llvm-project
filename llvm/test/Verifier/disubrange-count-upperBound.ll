; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

!named = !{!0}
; CHECK: Subrange can have any one of count or upperBound
!0 = !DISubrange(count: 20, lowerBound: 1, upperBound: 10)
