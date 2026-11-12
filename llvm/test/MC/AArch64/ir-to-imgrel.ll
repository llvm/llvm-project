; RUN: llc -mtriple=aarch64-pc-win32 %s -o - | FileCheck %s --check-prefix=AARCH64

@__ImageBase = external global i8

; AARCH64: .xword   "?x@@3HA"@IMGREL
@"\01?x@@3HA" = global i64 sub nsw (i64 ptrtoint (ptr @"\01?x@@3HA" to i64), i64 ptrtoint (ptr @__ImageBase to i64)), align 8

declare void @f()

; AARCH64: .xword   f@IMGREL
@fp = global i64 sub nsw (i64 ptrtoint (ptr @f to i64), i64 ptrtoint (ptr @__ImageBase to i64)), align 8
