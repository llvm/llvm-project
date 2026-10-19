; RUN: llc -mtriple=x86_64-pc-win32 %s -o - | FileCheck %s --implicit-check-not="@IMGREL"
; RUN: not llc -mtriple=x86_64-pc-win32 -filetype=obj %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

; Test that we reject forms of aliases that are not direct aliases to a GlobalObject
; (e.g. aliases with GEP/offsets, chained aliases, or aliases to thread-local globals),
; as well as non-dso_local targets (such as dllimport).
; When rejected, lowerRelativeReference returns nullptr, so the reference falls back
; to a raw subtraction (symbol - __ImageBase) in text assembly, and fails object emission
; because __ImageBase cannot be undefined in a subtraction expression in COFF.

@__ImageBase = external global i8

@target = internal global i32 42
@alias = hidden alias i32, ptr @target

@alias_gep = hidden alias i32, getelementptr (i32, ptr @target, i32 1)
@alias_to_alias = hidden alias i32, ptr @alias

@target_tls = thread_local global i32 42
@alias_tls = hidden alias i32, ptr @target_tls

@imported_var = external dllimport global i32

; Rejected: aliasee has an offset (GEP)
; CHECK:      alias_gep_ref:
; CHECK-NEXT:   .long alias_gep-__ImageBase
@alias_gep_ref = global i32 trunc (i64 sub nsw (i64 ptrtoint (ptr @alias_gep to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), align 4

; Rejected: aliasee is another alias (chained alias)
; CHECK:      alias_to_alias_ref:
; CHECK-NEXT:   .long alias_to_alias-__ImageBase
@alias_to_alias_ref = global i32 trunc (i64 sub nsw (i64 ptrtoint (ptr @alias_to_alias to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), align 4

; Rejected: aliasee is thread-local
; CHECK:      alias_tls_ref:
; CHECK-NEXT:   .long alias_tls-__ImageBase
@alias_tls_ref = global i32 trunc (i64 sub nsw (i64 ptrtoint (ptr @alias_tls to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), align 4

; Rejected: dllimport variable is not dso_local
; CHECK:      imported_var_ref:
; CHECK-NEXT:   .long imported_var-__ImageBase
@imported_var_ref = global i32 trunc (i64 sub nsw (i64 ptrtoint (ptr @imported_var to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), align 4

; ERR: error: symbol '__ImageBase' can not be undefined in a subtraction expression

