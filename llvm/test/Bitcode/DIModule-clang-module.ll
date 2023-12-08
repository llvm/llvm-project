; RUN: llvm-dis -o - %s.bc | FileCheck %s

; CHECK: DIModule(scope: null, name: "DebugModule", configMacros: "-DMODULES=0", includePath: "/", apinotes: "m.apinotes")

; ModuleID = 'DIModule-clang-module.ll'

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux"

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_ObjC_plus_plus, file: !1, producer: "clang version 11.0.0", isOptimized: false, runtimeVersion: 2, emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !2, imports: !3)
!1 = !DIFile(filename: "/test.cpp", directory: "/")
!2 = !{}
!3 = !{!4}
!4 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !5, file: !1, line: 5)
!5 = !DIModule(scope: null, name: "DebugModule", configMacros: "-DMODULES=0", includePath: "/", apinotes: "m.apinotes")
!6 = !{i32 2, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{!"clang version 11.0.0"}
