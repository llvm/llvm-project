; Verify that a COFF comdat whose leader symbol is absent is diagnosed with a
; clean error rather than a report_fatal_error crash in the backend.
; RUN: not llc -mtriple=x86_64-pc-win32 %s -o /dev/null 2>&1 | FileCheck %s

$missing_leader = comdat any
@assoc = global i32 0, comdat($missing_leader)

; CHECK: Associative COMDAT symbol 'missing_leader' does not exist
