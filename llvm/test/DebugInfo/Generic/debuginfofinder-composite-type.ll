; RUN: opt -passes='print<module-debuginfo>' -disable-output 2>&1 < %s \
; RUN:   | FileCheck %s

; This is to track DebugInfoFinder's ability to find the debug info metadata,
; in particular, properly visit DICompositeType slots.

; CHECK: Compile unit: DW_LANG_C_plus_plus from /somewhere/source.cpp
; CHECK: Subprogram: foo from /somewhere/source.cpp:1 ('_Z3foov')
; CHECK: Subprogram: bar from /somewhere/source.cpp:5
; CHECK: Subprogram: imported from /somewhere/source.cpp:3
; CHECK: Type: DW_TAG_subroutine_type
; CHECK: Type: T from /somewhere/source.cpp:2 DW_TAG_structure_type
; CHECK: Type: int_for_vtable DW_ATE_signed
; CHECK: Type: x from /somewhere/source.cpp:3 DW_TAG_member
; CHECK: Type: int DW_ATE_signed
; CHECK: Type: array_type DW_TAG_array_type
; CHECK: Type: sr__int_range DW_TAG_subrange_type
; CHECK: Type: int_subrange_type_base DW_ATE_signed
; CHECK: Type: int_for_subrange_type_lower DW_ATE_signed
; CHECK: Type: int_for_subrange_type_upper DW_ATE_signed
; CHECK: Type: int_for_subrange_type_stride DW_ATE_signed
; CHECK: Type: int_for_bias DW_ATE_signed
; CHECK: Type: int_for_subrange_lower DW_ATE_signed
; CHECK: Type: int_for_subrange_upper DW_ATE_signed
; CHECK: Type: int_for_subrange_stride DW_ATE_signed
; CHECK: Type: int_for_subrange_count DW_ATE_signed

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
!7 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 1, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !8, type: !23)
!8 = !{!9, !15, !19}
!9 = !DILocalVariable(name: "v", scope: !7, file: !1, line: 8, type: !10)
!10 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "T", scope: !7, file: !1, line: 2, size: 32, flags: DIFlagTypePassByValue, elements: !11, vtableHolder: !18)
!11 = !{!12, !14}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !10, file: !1, line: 3, baseType: !13, size: 32)
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !DISubprogram(name: "bar", scope: !10, file: !1, line: 5, scopeLine: 5, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: DISPFlagLocalToUnit, type: !23)
!15 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !7, entity: !16, file: !1, line: 7)
!16 = distinct !DISubprogram(name: "imported", scope: !17, file: !1, line: 3, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, type: !23)
!17 = !DINamespace(name: "ns", scope: null)
!18 = !DIBasicType(name: "int_for_vtable", size: 32, encoding: DW_ATE_signed)
!19 = !DILocalVariable(name: "array_var", scope: !7, file: !1, line: 8, type: !20)
!20 = !DICompositeType(name: "array_type", tag: DW_TAG_array_type, size: 64, align: 32, baseType: !13, elements: !21)
!21 = !{!22, !25, !41}
!22 = !DISubrangeType(name: "sr__int_range", size: 32, align: 32, baseType: !34, lowerBound: !35, upperBound: !37, stride: !39, bias: !32)
!23 = !DISubroutineType(types: !24)
!24 = !{null}
!25 = !DISubrange(lowerBound: !26, upperBound: !28, stride: !30)
!26 = !DILocalVariable(name: "var_for_subrange_lower", scope: !7, file: !1, line: 8, type: !27)
!27 = !DIBasicType(name: "int_for_subrange_lower", size: 32, encoding: DW_ATE_signed)
!28 = !DILocalVariable(name: "var_for_subrange_upper", scope: !7, file: !1, line: 8, type: !29)
!29 = !DIBasicType(name: "int_for_subrange_upper", size: 32, encoding: DW_ATE_signed)
!30 = !DILocalVariable(name: "var_for_subrange_stride", scope: !7, file: !1, line: 8, type: !31)
!31 = !DIBasicType(name: "int_for_subrange_stride", size: 32, encoding: DW_ATE_signed)
!32 = !DILocalVariable(name: "var_for_bias", scope: !7, file: !1, line: 8, type: !33)
!33 = !DIBasicType(name: "int_for_bias", size: 32, encoding: DW_ATE_signed)
!34 = !DIBasicType(name: "int_subrange_type_base", size: 32, encoding: DW_ATE_signed)
!35 = !DILocalVariable(name: "var_for_subrange_type_lower", scope: !7, file: !1, line: 8, type: !36)
!36 = !DIBasicType(name: "int_for_subrange_type_lower", size: 32, encoding: DW_ATE_signed)
!37 = !DILocalVariable(name: "var_for_subrange_type_upper", scope: !7, file: !1, line: 8, type: !38)
!38 = !DIBasicType(name: "int_for_subrange_type_upper", size: 32, encoding: DW_ATE_signed)
!39 = !DILocalVariable(name: "var_for_subrange_type_stride", scope: !7, file: !1, line: 8, type: !40)
!40 = !DIBasicType(name: "int_for_subrange_type_stride", size: 32, encoding: DW_ATE_signed)
!41 = !DISubrange(count: !42)
!42 = !DILocalVariable(name: "var_for_subrange_count", scope: !7, file: !1, line: 8, type: !43)
!43 = !DIBasicType(name: "int_for_subrange_count", size: 32, encoding: DW_ATE_signed)
