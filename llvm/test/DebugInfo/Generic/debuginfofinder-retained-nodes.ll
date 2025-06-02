; RUN: opt -passes='print<module-debuginfo>' -disable-output 2>&1 < %s \
; RUN:   | FileCheck %s

; This is to track DebugInfoFinder's ability to find the debug info metadata,
; in particular, properly visit DISubprogram's retainedNodes.

; CHECK: Compile unit: DW_LANG_C_plus_plus from /somewhere/source.cpp
; CHECK: Subprogram: foo from /somewhere/source.cpp:1 ('_Z3foov')
; CHECK: Subprogram: bar from /somewhere/source.cpp:5
; CHECK: Subprogram: imported from /somewhere/source.cpp:3
; CHECK: Type: T from /somewhere/source.cpp:2 DW_TAG_structure_type

%struct.T = type { i32 }

; Function Attrs: mustprogress noinline nounwind optnone ssp uwtable(sync)
define noundef i32 @_Z3foov() !dbg !7 {
entry:
  ret i32 0
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 21.0.0git", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "source.cpp", directory: "/somewhere")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{!"clang version 21.0.0git"}
!7 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 1, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !8)
!8 = !{!9, !15}
!9 = !DILocalVariable(name: "v", scope: !7, file: !1, line: 8, type: !10)
!10 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "T", scope: !7, file: !1, line: 2, size: 32, flags: DIFlagTypePassByValue, elements: !11)
!11 = !{!12, !14}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !10, file: !1, line: 3, baseType: !13, size: 32)
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !DISubprogram(name: "bar", scope: !10, file: !1, line: 5, scopeLine: 5, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: DISPFlagLocalToUnit)
!15 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !7, entity: !16, file: !1, line: 7)
!16 = distinct !DISubprogram(name: "imported", scope: !17, file: !1, line: 3, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!17 = !DINamespace(name: "ns", scope: null)