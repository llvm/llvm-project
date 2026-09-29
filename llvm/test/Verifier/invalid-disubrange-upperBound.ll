; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

!named = !{!0, !1}
; CHECK: UpperBound must be signed constant or DIVariable or DIExpression
!0 = !DISubrange(lowerBound: 1, upperBound: !1)
!1 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
