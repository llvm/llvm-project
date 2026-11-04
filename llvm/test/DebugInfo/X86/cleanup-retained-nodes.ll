; RUN: llvm-as %s -o %t.bc
; RUN: llvm-as %p/../Inputs/cleanup-retained-nodes.ll -o %t.global.bc
; RUN: llvm-link %t.global.bc %t.bc %t.bc -o - | llvm-dis - -o - \
; RUN:    | FileCheck %s --implicit-check-not=DICompositeType

; During module loading, if a local type appears in retainedNodes
; field of multiple DISubprograms due to ODR-uniquing,
; retainedNodes should be cleaned up, so that only one DISubprogram
; will have this type in its retainedNodes.

; CHECK: distinct !DICompositeType(tag: DW_TAG_class_type, {{.*}}, identifier: "type_global_in_another_module")
; CHECK: [[EMPTY:![0-9]+]] = !{}
; CHECK: [[BAR1:![0-9]+]] = distinct !DISubprogram(name: "bar", {{.*}}, retainedNodes: [[RN_BAR1:![0-9]+]])
; CHECK: [[RN_BAR1]] = !{[[T1:![0-9]+]], [[T1]], [[T1]], [[T2:![0-9]+]]}
; CHECK: [[T1]] = distinct !DICompositeType(tag: DW_TAG_class_type, scope: [[BAR1]], {{.*}}, identifier: "local_type")
; CHECK: [[T2]] = distinct !DICompositeType(tag: DW_TAG_class_type, scope: [[LB:![0-9]+]], {{.*}}, identifier: "local_type_in_block")
; CHECK: [[LB]] = !DILexicalBlock(scope: [[BAR1]]
; CHECK: {{![0-9]+}} = distinct !DISubprogram(name: "bar", {{.*}}, retainedNodes: [[EMPTY]])

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@llvm.used = appending global [1 x ptr] [ptr @bar]

define internal void @bar(ptr %this) !dbg !10 {
  ret void
}

!llvm.module.flags = !{!0, !1, !2}
!llvm.dbg.cu = !{!8}

!0 = !{i32 7, !"Dwarf Version", i32 2}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = !{i32 1, !"wchar_size", i32 4}
!3 = !DICompositeType(tag: DW_TAG_class_type, scope: !10, file: !9, line: 212, size: 8, flags: DIFlagTypePassByValue, elements: !7, identifier: "type_global_in_another_module")
!4 = !DICompositeType(tag: DW_TAG_class_type, scope: !5, file: !9, line: 211, size: 8, flags: DIFlagTypePassByValue, elements: !7, identifier: "local_type_in_block")
!5 = !DILexicalBlock(scope: !10)
; All repeating occurences of a uniqued type in retainedNodes must be checked.
!6 = !{!12, !12, !12, !4, !3}
!7 = !{}
!8 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !9, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!9 = !DIFile(filename: "tmp.cpp", directory: "/tmp/")
!10 = distinct !DISubprogram(name: "bar", scope: !9, file: !9, line: 68, type: !11, scopeLine: 68, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !8, retainedNodes: !6)
!11 = !DISubroutineType(types: !7)
!12 = !DICompositeType(tag: DW_TAG_class_type, scope: !10, file: !9, line: 210, size: 8, flags: DIFlagTypePassByValue, elements: !7, identifier: "local_type")
