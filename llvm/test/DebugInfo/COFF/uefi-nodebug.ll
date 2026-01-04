; RUN: llc -filetype=obj -o - < %s | llvm-readobj --codeview - | FileCheck %s
; Check that compiler info is not emitted when CodeView flag is not specified

; CHECK-NOT:  CodeViewTypes
; CHECK-NOT:  CodeViewDebugInfo

source_filename = "empty"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-uefi"

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "clang", emissionKind: NoDebug)
!1 = !DIFile(filename: "empty", directory: "path/to")
!2 = !{i32 2, !"Debug Info Version", i32 3}
