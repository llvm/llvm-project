; RUN: opt -passes='print<module-debuginfo>' -disable-output 2>&1 < %s \
; RUN:   | FileCheck %s

; This is to track DebugInfoFinder's ability to find the debug info metadata,
; in particular, properly visit different kinds of DIImportedEntities.

; Derived from the following C++ snippet
;
; namespace s {
;   int i;
; }
;
; using s::i;
;
; compiled with `clang -O1 -g3 -emit-llvm -S`

; CHECK: Compile unit: DW_LANG_C_plus_plus from /somewhere/source.cpp
; CHECK: Global variable: i from /somewhere/source.cpp:2 ('_ZN1s1iE')
; CHECK: Type: int DW_ATE_signed

@_ZN1s1iE = local_unnamed_addr global i32 0, align 4, !dbg !0

!llvm.dbg.cu = !{!5}
!llvm.module.flags = !{!10, !11, !12, !13}
!llvm.ident = !{!14}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "i", linkageName: "_ZN1s1iE", scope: !2, file: !3, line: 2, type: !4, isLocal: false, isDefinition: true)
!2 = !DINamespace(name: "s", scope: null)
!3 = !DIFile(filename: "source.cpp", directory: "/somewhere")
!4 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 7.0.99", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !6, globals: !7, imports: !8)
!6 = !{}
!7 = !{!0}
!8 = !{!9}
!9 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !1, file: !3, line: 5)
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{i32 7, !"PIC Level", i32 2}
!14 = !{!"clang version 7.0.99"}
