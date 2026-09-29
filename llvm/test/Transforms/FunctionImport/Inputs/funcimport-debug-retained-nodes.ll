target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @func() {
entry:
  ret void, !dbg !8
}

define i32 @foo() !dbg !13 {
entry:
  ret i32 0, !dbg !14
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "funcimport_debug2.c", directory: ".")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "func", scope: !1, file: !1, line: 2, type: !6, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = !DILocation(line: 2, column: 1, scope: !9, inlinedAt: !12)
!9 = distinct !DISubprogram(name: "inlined_out_clone", scope: !1, file: !1, line: 20, type: !6, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !10)
!10 = !{!11}
!11 = !DICompositeType(tag: DW_TAG_class_type, scope: !9, file: !1, line: 210, size: 8, flags: DIFlagTypePassByValue, elements: !2, identifier: "local_type")
!12 = !DILocation(line: 11, scope: !5)
!13 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!14 = !DILocation(line: 3, column: 1, scope: !9, inlinedAt: !15)
!15 = !DILocation(line: 3, column: 1, scope: !13)
