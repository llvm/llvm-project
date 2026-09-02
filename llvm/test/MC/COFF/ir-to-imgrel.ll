; RUN: llc -mtriple=x86_64-pc-win32 %s -o - | FileCheck %s --check-prefix=X64

@__ImageBase = external global i8

; X64: .quad   "?x@@3HA"@IMGREL
@"\01?x@@3HA" = global i64 sub nsw (i64 ptrtoint (ptr @"\01?x@@3HA" to i64), i64 ptrtoint (ptr @__ImageBase to i64)), align 8

declare void @f()

; X64: .quad   f@IMGREL
@fp = global i64 sub nsw (i64 ptrtoint (ptr @f to i64), i64 ptrtoint (ptr @__ImageBase to i64)), align 8

@target = internal global i32 42
@alias = hidden alias i32, ptr @target

; X64: .quad   alias@IMGREL
@alias_ref = global i64 sub nsw (i64 ptrtoint (ptr @alias to i64), i64 ptrtoint (ptr @__ImageBase to i64)), align 8

