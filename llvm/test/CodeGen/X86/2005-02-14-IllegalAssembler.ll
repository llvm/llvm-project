; RUN: llc < %s -mtriple=i686-- | FileCheck %s

@A = external global i32                ; <ptr> [#uses=1]
@Y = global ptr getelementptr (i32, ptr @A, i32 -1)                ; <ptr> [#uses=0]
; CHECK-NOT: 18446744073709551612

