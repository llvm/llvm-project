; RUN: not --crash llc < %s -mtriple=i386-linux 2>&1 | FileCheck %s

; ptrtoint expressions that cast to a wider integer type are not supported.
; A frontend can achieve a similar result by casting to the correct integer
; type and explicitly zeroing any additional bytes.
; { i32, i32 } { i32 ptrtoint (ptr @r to i32), i32 0 }

; CHECK: LLVM ERROR: Unsupported expression in static initializer: ptrtoint (ptr @r to i64)

@r = global i64 ptrtoint (ptr @r to i64)
