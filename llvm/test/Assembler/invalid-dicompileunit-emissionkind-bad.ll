; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

!llvm.dbg.cu = !{!0}
; CHECK: <stdin>:[[@LINE+1]]:71: error: value for 'emissionKind' too large
!0 = distinct !DICompileUnit(language: DW_LANG_Cobol85, emissionKind: 99,
                             file: !DIFile(filename: "a", directory: "b"))
!llvm.module.flags = !{!1}
!1 = !{i32 2, !"Debug Info Version", i32 3}
