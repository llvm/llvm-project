; Test to ensure DICompositeType are imported as type declarations
; for ThinLTO

; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/debuginfo-compositetype-import.ll -o %t2.bc
; RUN: llvm-lto -thinlto-action=thinlink -o %t.index.bc %t1.bc %t2.bc

; By default, composite types are imported as type declarations
; RUN: llvm-lto -thinlto-action=import %t2.bc -thinlto-index=%t.index.bc -o - | llvm-dis -o - | FileCheck %s
; RUN: llvm-lto2 run %t1.bc %t2.bc -o %t.out -save-temps \
; RUN:   -r %t2.bc,main,plx \
; RUN:   -r %t2.bc,foo,l \
; RUN:   -r %t1.bc,foo,pl
; RUN: llvm-dis < %t.out.2.3.import.bc | FileCheck %s

; CHECK: distinct !DICompositeType(tag: DW_TAG_enumeration_type, name: "enum", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: 50, size: 32, flags: DIFlagFwdDecl, identifier: "enum")
; CHECK: distinct !DICompositeType(tag: DW_TAG_class_type, name: "class<>", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: 728, size: 448, flags: DIFlagFwdDecl, identifier: "class")
; CHECK: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "struct", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: 309, size: 128, flags: DIFlagFwdDecl, templateParams: !{{[0-9]+}}, identifier: "struct_templ_simplified")
; CHECK: distinct !DICompositeType(tag: DW_TAG_union_type, file: !{{[0-9]+}}, line: 115, size: 384, flags: DIFlagFwdDecl, identifier: "union")
; CHECK: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "struct<>", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: 309, size: 128, flags: DIFlagFwdDecl, identifier: "struct_templ")
; CHECK: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "_STNstruct|<>", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: 309, size: 128, flags: DIFlagFwdDecl, templateParams: !{{[0-9]+}}, identifier: "struct_templ_simplified_mangled")

; Ensure that full type definitions of composite types are imported if requested
; RUN: llvm-lto -import-full-type-definitions -thinlto-action=import %t2.bc -thinlto-index=%t.index.bc -o - | llvm-dis -o - | FileCheck %s --check-prefix=FULL
; RUN: llvm-lto2 run %t1.bc %t2.bc -o %t.out -save-temps \
; RUN:   -import-full-type-definitions \
; RUN:   -r %t2.bc,main,plx \
; RUN:   -r %t2.bc,foo,l \
; RUN:   -r %t1.bc,foo,pl
; RUN: llvm-dis < %t.out.2.3.import.bc | FileCheck %s --check-prefix=FULL

; FULL: distinct !DICompositeType(tag: DW_TAG_enumeration_type, name: "enum", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: 50, size: 32, elements: !{{[0-9]+}}, identifier: "enum")
; FULL: distinct !DICompositeType(tag: DW_TAG_class_type, name: "class<>", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: 728, size: 448, elements: !{{[0-9]+}}, identifier: "class")
; FULL: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "struct", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: 309, baseType: !{{[0-9]+}}, size: 128, offset: 64, elements: !{{[0-9]+}}, vtableHolder: !{{[0-9]+}}, templateParams: !{{[0-9]+}}, identifier: "struct_templ_simplified")
; FULL: distinct !DICompositeType(tag: DW_TAG_union_type, file: !{{[0-9]+}}, line: 115, size: 384, elements: !{{[0-9]+}}, identifier: "union")
; FULL: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "struct<>", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: 309, baseType: !{{[0-9]+}}, size: 128, offset: 64, elements: !{{[0-9]+}}, vtableHolder: !{{[0-9]+}}, templateParams: !{{[0-9]+}}, identifier: "struct_templ")
; FULL: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "_STNstruct|<>", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: 309, baseType: !{{[0-9]+}}, size: 128, offset: 64, elements: !{{[0-9]+}}, vtableHolder: !{{[0-9]+}}, templateParams: !{{[0-9]+}}, identifier: "struct_templ_simplified_mangled")

; ModuleID = 'debuginfo-compositetype-import.c'
source_filename = "debuginfo-compositetype-import.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define void @foo() #0 !dbg !6 {
entry:
    ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!llvm.ident = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 4.0.0 (trunk 286863) (llvm/trunk 286875)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "debuginfo-compositetype-import.c", directory: "")
!2 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{!"clang version 4.0.0 (trunk 286863) (llvm/trunk 286875)"}
!5 = !{}
!6 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !7, isLocal: false, isDefinition: true, scopeLine: 2, isOptimized: false, unit: !0, retainedNodes: !5)
!7 = !DISubroutineType(types: !8)
!8 = !{!9, !10, !11, !12, !13, !14}
!9 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "enum", scope: !1, file: !1, line: 50, size: 32, elements: !5, identifier: "enum")
!10 = !DICompositeType(tag: DW_TAG_class_type, name: "class<>", scope: !1, file: !1, line: 728, size: 448, elements: !5, identifier: "class")
!11 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "struct", scope: !1, file: !1, line: 309, baseType: !10, size: 128, offset: 64, elements: !5, vtableHolder: !10, templateParams: !5, identifier: "struct_templ_simplified")
!12 = distinct !DICompositeType(tag: DW_TAG_union_type, file: !1, line: 115, size: 384, elements: !5, identifier: "union")
!13 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "struct<>", scope: !1, file: !1, line: 309, baseType: !10, size: 128, offset: 64, elements: !5, vtableHolder: !10, templateParams: !5, identifier: "struct_templ")
!14 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "_STNstruct|<>", scope: !1, file: !1, line: 309, baseType: !10, size: 128, offset: 64, elements: !5, vtableHolder: !10, templateParams: !5, identifier: "struct_templ_simplified_mangled")
