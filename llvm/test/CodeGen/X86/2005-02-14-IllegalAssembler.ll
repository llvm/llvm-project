; RUN: llc < %s -march=x86 | not grep 18446744073709551612

@A = external global i32                ; <i32*> [#uses=1]
@Y = global i32* getelementptr (i32* @A, i32 -1)                ; <i32**> [#uses=0]

