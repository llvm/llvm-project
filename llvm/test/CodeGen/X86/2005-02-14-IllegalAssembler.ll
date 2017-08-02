; RUN: llc < %s -mtriple=i686-- | FileCheck %s

@A = external global i32                ; <i32*> [#uses=1]
@Y = global i32* getelementptr (i32, i32* @A, i32 -1)                ; <i32**> [#uses=0]
; CHECK-NOT: 18446744073709551612

