; RUN: %llc_dwarf -O1 -filetype=obj -split-dwarf-file=%t.dwo < %s | llvm-dwarfdump -debug-info - | FileCheck %s --implicit-check-not "{{DW_TAG|NULL}}"

; CHECK-LABEL: debug_info contents
; CHECK: DW_TAG_compile_unit
; CHECK:   DW_AT_GNU_dwo_name
; CHECK:   DW_AT_GNU_dwo_id
; CHECK:   DW_TAG_subprogram
; CHECK:   DW_TAG_subprogram
; CHECK:     DW_TAG_inlined_subroutine
; CHECK:     NULL
; CHECK:   NULL

; CHECK-LABEL: debug_info.dwo contents

; CHECK: DW_TAG_compile_unit
; CHECK:   DW_TAG_subprogram
; CHECK:     DW_TAG_imported_declaration
; CHECK:     NULL
; CHECK:   DW_TAG_subprogram
; CHECK:     DW_TAG_inlined_subroutine
; CHECK:     NULL
; CHECK:   DW_TAG_namespace
; CHECK:     DW_TAG_structure_type
; CHECK:     NULL
; CHECK:   DW_TAG_base_type
; CHECK:   NULL

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"struct.ns::t1" = type { i8 }

declare void @llvm.dbg.declare(metadata, metadata, metadata)

declare dso_local void @_Z3pinv() local_unnamed_addr

define dso_local i32 @main() local_unnamed_addr !dbg !18 {
entry:
  call void @llvm.dbg.declare(metadata %"struct.ns::t1"* undef, metadata !22, metadata !DIExpression()), !dbg !23
  call void @_Z3pinv(), !dbg !25
  ret i32 0, !dbg !26
}

!llvm.dbg.cu = !{!0, !10}
!llvm.ident = !{!12, !12}
!llvm.module.flags = !{!13, !14, !15, !16, !17}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 14.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: true, nameTableKind: GNU)
!1 = !DIFile(filename: "a.cpp", directory: "/")
!2 = !{!3}
!3 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !8, file: !1, line: 3)
!4 = distinct !DISubprogram(name: "f1", linkageName: "_Z2f1v", scope: !1, file: !1, line: 3, type: !5, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !{}
!8 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t1", scope: !9, file: !1, line: 1, size: 8, flags: DIFlagTypePassByValue, elements: !7, identifier: "_ZTSN2ns2t1E")
!9 = !DINamespace(name: "ns", scope: null)
!10 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !11, producer: "clang version 14.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: true, nameTableKind: GNU)
!11 = !DIFile(filename: "b.cpp", directory: "/")
!12 = !{!"clang version 14.0.0"}
!13 = !{i32 7, !"Dwarf Version", i32 4}
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{i32 1, !"wchar_size", i32 4}
!16 = !{i32 7, !"uwtable", i32 1}
!17 = !{i32 7, !"frame-pointer", i32 2}
!18 = distinct !DISubprogram(name: "main", scope: !11, file: !11, line: 2, type: !19, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !10, retainedNodes: !7)
!19 = !DISubroutineType(types: !20)
!20 = !{!21}
!21 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!22 = !DILocalVariable(name: "v1", scope: !4, file: !1, line: 3, type: !8)
!23 = !DILocation(line: 3, column: 37, scope: !4, inlinedAt: !24)
!24 = distinct !DILocation(line: 3, column: 3, scope: !18)
!25 = !DILocation(line: 3, column: 41, scope: !4, inlinedAt: !24)
!26 = !DILocation(line: 4, column: 1, scope: !18)
