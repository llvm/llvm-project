; RUN: opt -check-debugify < %s -S -o - 2>&1 | FileCheck %s -implicit-check-not "WARNING: Instruction with empty DebugLoc in function bar"

; CHECK: WARNING: Instruction with empty DebugLoc in function foo --   ret void
define void @foo() !dbg !6 {
  ret void
}

define i32 @bar() !dbg !9 {
  ret i32 0, !dbg !15
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!3, !4}
!llvm.module.flags = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "void", directory: "/")
!2 = !{}
!3 = !{i32 6}
!4 = !{i32 1}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: null, file: !1, line: 1, type: !7, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: true, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !2)
!9 = distinct !DISubprogram(name: "bar", linkageName: "bar", scope: null, file: !1, line: 2, type: !7, isLocal: false, isDefinition: true, scopeLine: 2, isOptimized: true, unit: !0, retainedNodes: !10)
!10 = !{!11}
!11 = !DILocalVariable(name: "1", scope: !9, file: !1, line: 3, type: !12)
!12 = !DIBasicType(name: "ty32", size: 32, encoding: DW_ATE_unsigned)
!15 = !DILocation(line: 0, column: 1, scope: !9)
