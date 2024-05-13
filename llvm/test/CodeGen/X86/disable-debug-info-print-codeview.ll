; RUN: llc -disable-debug-info-print -o - %s | FileCheck %s

; Check that debug info isn't emitted for CodeView with
; -disable-debug-info-print.

; CHECK-NOT:      CodeViewTypes
; CHECK-NOT:      CodeViewDebugInfo

source_filename = "empty"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.24215"

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "clang", emissionKind: FullDebug)
!1 = !DIFile(filename: "empty", directory: "path/to")
!2 = !{i32 2, !"CodeView", i32 1}
!3 = !{i32 2, !"Debug Info Version", i32 3}
