; RUN: not --crash llc -mtriple=i686-linux-gnu < %s 2> %t
; RUN: FileCheck --check-prefix=CHECK-ERRORS < %t %s

@0 = global i8 extractvalue ([1 x i8] select (i1 ptrtoint (ptr @1 to i1), [1 x i8] [ i8 1 ], [1 x i8] [ i8 2 ]), 0)
@1 = external global i32

; CHECK-ERRORS: Unsupported expression in static initializer: extractvalue

