; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s

; CHECK: !{{.*}} = distinct !DICompileUnit(language: DW_LANG_C_plus_plus
; CHECK-SAME: dialect: DW_LLVM_LANG_DIALECT_simt
; CHECK: !{{.*}} = distinct !DICompileUnit(sourceLanguageName: DW_LNAME_C_plus_plus, sourceLanguageVersion: 202002
; CHECK-SAME: dialect: DW_LLVM_LANG_DIALECT_tile

source_filename = "cu.cpp"
target triple = "x86_64-unknown-linux-gnu"

!llvm.dbg.cu = !{!0, !1}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, globals: !5, splitDebugInlining: false, nameTableKind: None, dialect: DW_LLVM_LANG_DIALECT_simt)
!1 = distinct !DICompileUnit(sourceLanguageName: DW_LNAME_C_plus_plus, sourceLanguageVersion: 202002, file: !6, producer: "", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, globals: !5, splitDebugInlining: false, nameTableKind: None, dialect: DW_LLVM_LANG_DIALECT_tile)
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DIFile(filename: "cu.cpp", directory: "/tmp")
!5 = !{}
!6 = !DIFile(filename: "cu2.cpp", directory: "/tmp")
