; RUN: llc --filetype=obj %s -o - | dxil-dis -o - | FileCheck %s

target triple = "dxil-unknown-shadermodel6.3-library"

;; CHECK: distinct !DICompileUnit(language: DW_LANG_C99,

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(sourceLanguageName: DW_LNAME_C, sourceLanguageVersion: 199901, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "versioned-language.c", directory: "")
!2 = !{i32 7, !"Dwarf Version", i32 6}
!3 = !{i32 2, !"Debug Info Version", i32 3}
