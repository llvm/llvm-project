; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

!named = !{!0, !1, !2}
; CHECK: LowerBound must be signed constant or DIVariable or DIExpression
!0 = !DISubrange(count: 5, lowerBound: i64 poison)
; CHECK: UpperBound must be signed constant or DIVariable or DIExpression
!1 = !DISubrange(lowerBound: 0, upperBound: i64 poison)
; CHECK: Stride must be signed constant or DIVariable or DIExpression
!2 = !DISubrange(count: 5, stride: i64 poison)
