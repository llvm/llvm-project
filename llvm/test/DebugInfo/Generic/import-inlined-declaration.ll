; RUN: %llc_dwarf -O0 -filetype=obj < %s | llvm-dwarfdump - | FileCheck --implicit-check-not "{{DW_TAG|NULL}}" %s

; namespace ns {
; inline __attribute__((always_inline))
; void foo() { int a = 4; }
; }
;
; void goo() {
;   using ns::foo;
;   foo();
; }

; Ensure that imported declarations reference the correct subprograms even if
; those subprograms are inlined.

; CHECK: DW_TAG_compile_unit
; CHECK:   DW_TAG_namespace
; CHECK:     DW_AT_name     ("ns")
; CHECK: [[FOO:0x.*]]:     DW_TAG_subprogram
; CHECK:       DW_AT_name   ("foo")
; CHECK:       DW_TAG_variable
; CHECK:       NULL
; CHECK:     NULL
; CHECK:   DW_TAG_base_type
; CHECK:   DW_TAG_subprogram
; CHECK:     DW_AT_name     ("goo")
; CHECK:     DW_TAG_inlined_subroutine
; CHECK:       DW_AT_abstract_origin ([[FOO]]
; CHECK:       DW_TAG_variable
; CHECK:       NULL
; CHECK:     DW_TAG_imported_declaration
; CHECK:       DW_AT_import ([[FOO]])
; CHECK:     NULL
; CHECK:   NULL

; Function Attrs: mustprogress noinline optnone uwtable
define dso_local void @_Z3goov() !dbg !4 {
entry:
  %a.i = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %a.i, metadata !16, metadata !DIExpression()), !dbg !18
  store i32 4, i32* %a.i, align 4, !dbg !18
  ret void, !dbg !20
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11, !12, !13, !14}
!llvm.ident = !{!15}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 14.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "imported-inlined-declaration.cpp", directory: "")
!2 = !{!3}
!3 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !8, file: !1, line: 7)
!4 = distinct !DISubprogram(name: "goo", linkageName: "_Z3goov", scope: !1, file: !1, line: 6, type: !5, scopeLine: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !{}
!8 = distinct !DISubprogram(name: "foo", linkageName: "_ZN2ns3fooEv", scope: !9, file: !1, line: 3, type: !5, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !7)
!9 = !DINamespace(name: "ns", scope: null)
!10 = !{i32 7, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{i32 7, !"uwtable", i32 1}
!14 = !{i32 7, !"frame-pointer", i32 2}
!15 = !{!"clang version 14.0.0"}
!16 = !DILocalVariable(name: "a", scope: !8, file: !1, line: 3, type: !17)
!17 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!18 = !DILocation(line: 3, column: 18, scope: !8, inlinedAt: !19)
!19 = distinct !DILocation(line: 8, column: 2, scope: !4)
!20 = !DILocation(line: 9, column: 1, scope: !4)
