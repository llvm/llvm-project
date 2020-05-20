; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: DILocation not allowed within this metadata node
; CHECK-NEXT: [[unknownMD:![0-9]+]] = distinct !{[[unknownMD]], [[dbgMD:![0-9]+]]}
; CHECK-NEXT: [[dbgMD]] = !DILocation

define void @f() !dbg !5 {
  ret void, !dbg !10, !unknown_md !11
}

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!3, !3}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "loop.ll", directory: "/")
!2 = !{}
!3 = !{i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "f", linkageName: "f", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !7)
!6 = !DISubroutineType(types: !2)
!7 = !{!8}
!8 = !DILocalVariable(name: "1", scope: !5, file: !1, line: 1, type: !9)
!9 = !DIBasicType(name: "ty32", size: 32, encoding: DW_ATE_unsigned)
!10 = !DILocation(line: 1, column: 1, scope: !5)
!11 = !{!11, !10}
