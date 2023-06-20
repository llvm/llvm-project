; RUN: %llc_dwarf -O0 -filetype=obj < %s | llvm-dwarfdump -debug-info - | FileCheck --implicit-check-not "{{DW_TAG|NULL}}" %s
; REQUIRES: object-emission

; inline __attribute__((always_inline))
; int removed() { struct A {int i;}; struct A a; return a.i++; }
;
; __attribute__((always_inline))
; int not_removed() { struct B {int i;}; struct B b; return b.i++; }
;
; int foo() { return removed() + not_removed(); }}

; Ensure that function-local types have the correct subprogram parent even if
; those subprograms are inlined.

; CHECK: DW_TAG_compile_unit
; CHECK:   DW_TAG_subprogram
; CHECK:     DW_AT_abstract_origin	({{0x.*}} "not_removed")
; CHECK:     DW_TAG_variable
; CHECK:     NULL
; CHECK:   DW_TAG_subprogram
; CHECK:     DW_AT_name	("removed")
; CHECK: [[A:0x.*]]: DW_TAG_structure_type
; CHECK:       DW_AT_name	("A")
; CHECK:       DW_TAG_member
; CHECK:       NULL
; CHECK:     DW_TAG_variable
; CHECK:       DW_AT_type	([[A]] "A")
; CHECK:     NULL
; CHECK:   DW_TAG_base_type
; CHECK:   DW_TAG_subprogram
; CHECK:     DW_AT_name	("not_removed")
; CHECK: [[B:0x.*]]: DW_TAG_structure_type
; CHECK:       DW_AT_name	("B")
; CHECK:       DW_TAG_member
; CHECK:       NULL
; CHECK:     DW_TAG_variable
; CHECK:       DW_AT_type	([[B]] "B")
; CHECK:     NULL
; CHECK:   DW_TAG_subprogram
; CHECK:     DW_TAG_inlined_subroutine
; CHECK:       DW_TAG_variable
; CHECK:       NULL
; CHECK:     DW_TAG_inlined_subroutine
; CHECK:       DW_TAG_variable
; CHECK:       NULL
; CHECK:     NULL
; CHECK:   NULL

%struct.B = type { i32 }
%struct.A = type { i32 }

define dso_local i32 @not_removed() !dbg !12 {
  %1 = alloca %struct.B, align 4
  call void @llvm.dbg.declare(metadata %struct.B* %1, metadata !18, metadata !DIExpression()), !dbg !22
  %2 = getelementptr inbounds %struct.B, %struct.B* %1, i32 0, i32 0, !dbg !23
  %3 = load i32, i32* %2, align 4, !dbg !24
  %4 = add nsw i32 %3, 1, !dbg !24
  store i32 %4, i32* %2, align 4, !dbg !24
  ret i32 %3, !dbg !25
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)

define dso_local i32 @foo() !dbg !26 {
  %1 = alloca %struct.A, align 4
  %2 = alloca %struct.B, align 4
  call void @llvm.dbg.declare(metadata %struct.A* %1, metadata !27, metadata !DIExpression()), !dbg !32
  %3 = getelementptr inbounds %struct.A, %struct.A* %1, i32 0, i32 0, !dbg !34
  %4 = load i32, i32* %3, align 4, !dbg !35
  %5 = add nsw i32 %4, 1, !dbg !35
  store i32 %5, i32* %3, align 4, !dbg !35
  call void @llvm.dbg.declare(metadata %struct.B* %2, metadata !18, metadata !DIExpression()), !dbg !36
  %6 = getelementptr inbounds %struct.B, %struct.B* %2, i32 0, i32 0, !dbg !38
  %7 = load i32, i32* %6, align 4, !dbg !39
  %8 = add nsw i32 %7, 1, !dbg !39
  store i32 %8, i32* %6, align 4, !dbg !39
  %9 = add nsw i32 %4, %7, !dbg !40
  ret i32 %9, !dbg !41
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8, !9, !10}
!llvm.ident = !{!11}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 14.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "inlined-local-type.cpp", directory: "")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 1, !"branch-target-enforcement", i32 0}
!6 = !{i32 1, !"sign-return-address", i32 0}
!7 = !{i32 1, !"sign-return-address-all", i32 0}
!8 = !{i32 1, !"sign-return-address-with-bkey", i32 0}
!9 = !{i32 7, !"uwtable", i32 1}
!10 = !{i32 7, !"frame-pointer", i32 1}
!11 = !{!"clang version 14.0.0"}
!12 = distinct !DISubprogram(name: "not_removed", scope: !13, file: !13, line: 5, type: !14, scopeLine: 5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !43)
!13 = !DIFile(filename: "inlined-local-type.cpp", directory: "")
!14 = !DISubroutineType(types: !15)
!15 = !{!16}
!16 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!17 = !{}
!18 = !DILocalVariable(name: "b", scope: !12, file: !13, line: 5, type: !19)
!19 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "B", scope: !12, file: !13, line: 5, size: 32, elements: !20)
!20 = !{!21}
!21 = !DIDerivedType(tag: DW_TAG_member, name: "i", scope: !19, file: !13, line: 5, baseType: !16, size: 32)
!22 = !DILocation(line: 5, column: 49, scope: !12)
!23 = !DILocation(line: 5, column: 61, scope: !12)
!24 = !DILocation(line: 5, column: 62, scope: !12)
!25 = !DILocation(line: 5, column: 52, scope: !12)
!26 = distinct !DISubprogram(name: "foo", scope: !13, file: !13, line: 7, type: !14, scopeLine: 7, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !17)
!27 = !DILocalVariable(name: "a", scope: !28, file: !13, line: 2, type: !29)
!28 = distinct !DISubprogram(name: "removed", scope: !13, file: !13, line: 2, type: !14, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !42)
!29 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A", scope: !28, file: !13, line: 2, size: 32, elements: !30)
!30 = !{!31}
!31 = !DIDerivedType(tag: DW_TAG_member, name: "i", scope: !29, file: !13, line: 2, baseType: !16, size: 32)
!32 = !DILocation(line: 2, column: 45, scope: !28, inlinedAt: !33)
!33 = distinct !DILocation(line: 7, column: 20, scope: !26)
!34 = !DILocation(line: 2, column: 57, scope: !28, inlinedAt: !33)
!35 = !DILocation(line: 2, column: 58, scope: !28, inlinedAt: !33)
!36 = !DILocation(line: 5, column: 49, scope: !12, inlinedAt: !37)
!37 = distinct !DILocation(line: 7, column: 32, scope: !26)
!38 = !DILocation(line: 5, column: 61, scope: !12, inlinedAt: !37)
!39 = !DILocation(line: 5, column: 62, scope: !12, inlinedAt: !37)
!40 = !DILocation(line: 7, column: 30, scope: !26)
!41 = !DILocation(line: 7, column: 13, scope: !26)
!42 = !{!29}
!43 = !{!19}
