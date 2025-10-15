; REQUIRES: x86_64-linux
; RUN: %llc_dwarf -O1 -filetype=obj -split-dwarf-file=%t.dwo < %s \
; RUN:   | llvm-dwarfdump -debug-info -                           \
; RUN:   | FileCheck %s --implicit-check-not "{{DW_TAG|NULL}}"

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Ensure that the imported entity 'nn::A' gets emitted in 'foo()'s abstract tree
; in the destination (where 'foo()' was inlined) compile unit.

; CHECK-LABEL: .debug_info contents
; CHECK: DW_TAG_skeleton_unit
; CHECK:   DW_AT_dwo_name

; CHECK-LABEL: .debug_info.dwo contents:
; CHECK: DW_TAG_compile_unit
; CHECK:   DW_AT_name ("test.cpp")
; CHECK:   DW_AT_dwo_name

; CHECK:   [[ABSTRACT_FOO:0x[0-9a-f]+]]: DW_TAG_subprogram
; CHECK:     DW_AT_name	("foo")
; CHECK:     DW_TAG_imported_declaration
; CHECK:       DW_AT_import  ([[A:0x[0-9a-f]+]])
; CHECK:     NULL

; CHECK:   DW_TAG_base_type
; CHECK:     DW_AT_name	("int")

; CHECK:   DW_TAG_subprogram
; CHECK:     DW_AT_name	("main")
; CHECK:     DW_TAG_inlined_subroutine
; CHECK:       DW_AT_abstract_origin ([[ABSTRACT_FOO]] "_Z3foov")
; CHECK:     NULL

; CHECK:   DW_TAG_namespace
; CHECK:     DW_AT_name	("nn")
; CHECK:     [[A]]: DW_TAG_variable
; CHECK:       DW_AT_name	("A")
; CHECK:     NULL
; CHECK:   NULL

define dso_local noundef i32 @main() local_unnamed_addr !dbg !20 {
entry:
  ret i32 42, !dbg !21
}

!llvm.dbg.cu = !{!0, !2}
!llvm.module.flags = !{!13, !14}
!llvm.ident = !{!19, !19}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 15.0.0", isOptimized: true, runtimeVersion: 0, splitDebugFilename: "test.dwo", emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: GNU)
!1 = !DIFile(filename: "test.cpp", directory: "/", checksumkind: CSK_MD5, checksum: "e7c2808ee27614e496499d55e4b37962")
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 15.0.0", isOptimized: true, runtimeVersion: 0, splitDebugFilename: "cu1.dwo", emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: GNU)
!3 = !DIFile(filename: "cu1.cpp", directory: "/", checksumkind: CSK_MD5, checksum: "c0b84240ef5682b87083b33cf9038171")
!4 = !{!5}
!5 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !6, entity: !11, file: !3, line: 5)
!6 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !3, file: !3, line: 5, type: !7, scopeLine: 5, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!7 = !DISubroutineType(types: !8)
!8 = !{!9}
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !{}
!11 = distinct !DIGlobalVariable(name: "A", linkageName: "_ZN2nn1AE", scope: !12, file: !3, line: 2, type: !9, isLocal: false, isDefinition: true)
!12 = !DINamespace(name: "nn", scope: null)
!13 = !{i32 7, !"Dwarf Version", i32 5}
!14 = !{i32 2, !"Debug Info Version", i32 3}
!19 = !{!"clang version 15.0.0"}
!20 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 3, type: !7, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !10)
!21 = !DILocation(line: 4, column: 3, scope: !6, inlinedAt: !22)
!22 = !DILocation(line: 4, column: 3, scope: !20)
