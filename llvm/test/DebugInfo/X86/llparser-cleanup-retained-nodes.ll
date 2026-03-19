; RUN: llvm-link %s %s -S -o - | FileCheck %s --implicit-check-not=DICompositeType

; During module loading, if a local type appears in retainedNodes
; field of multiple DISubprograms due to ODR-uniquing,
; LLParser should clean up retainedNodes, so that only one DISubprogram
; will have this type in its retainedNodes.

; CHECK: [[BAR1:![0-9]+]] = distinct !DISubprogram(name: "bar", {{.*}}, retainedNodes: [[RN_BAR1:![0-9]+]])
; CHECK: [[EMPTY:![0-9]+]] = !{}
; CHECK: [[RN_BAR1]] = !{[[T1:![0-9]+]]}
; CHECK: [[T1]] = distinct !DICompositeType(tag: DW_TAG_class_type, scope: [[BAR1]], {{.*}}, identifier: "local_type")
; CHECK: {{![0-9]+}} = distinct !DISubprogram(name: "bar", {{.*}}, retainedNodes: [[EMPTY]])

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@llvm.used = appending global [1 x ptr] [ptr @bar]

define internal void @bar(ptr %this) !dbg !5 {
  ret void
}

!llvm.module.flags = !{!0, !1, !2}
!llvm.dbg.cu = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"Dwarf Version", i32 2}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !4, producer: "clang version 14.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!4 = !DIFile(filename: "tmp.cpp", directory: "/tmp/")
!5 = distinct !DISubprogram(name: "bar", scope: !4, file: !4, line: 68, type: !6, scopeLine: 68, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !3, retainedNodes: !9)
!6 = !DISubroutineType(types: !8)
!7 = !DICompositeType(tag: DW_TAG_class_type, scope: !5, file: !4, line: 210, size: 8, flags: DIFlagTypePassByValue, elements: !8, identifier: "local_type")
!8 = !{}
!9 = !{!7}
