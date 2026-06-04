; RUN: llc < %s -mtriple=s390x-ibm-zos | FileCheck %s

@fortytwo = hidden global i32 42, align 4, !dbg !0

define hidden signext i32 @getFortyTwo() !dbg !8 {
entry:
  %0 = load i32, ptr @fortytwo, align 4, !dbg !11
  ret i32 %0, !dbg !12
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!6, !7}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "fortytwo", scope: !2, file: !3, line: 2, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang version 22.0.0git", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "fortytwo.c", directory: "")
!4 = !{!0}
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !{i32 7, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = distinct !DISubprogram(name: "getFortyTwo", scope: !3, file: !3, line: 4, type: !9, scopeLine: 4, spFlags: DISPFlagDefinition, unit: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{!5}
!11 = !DILocation(line: 4, column: 28, scope: !8)
!12 = !DILocation(line: 4, column: 21, scope: !8)

; Check the emitted section definition
; CHECK: D_ABREV CATTR ALIGN(3),FILL(0),NOLOAD,RMODE(64)
; CHECK: D_INFO CATTR ALIGN(3),FILL(0),NOLOAD,RMODE(64)
; CHECK: D_STR CATTR ALIGN(3),FILL(0),NOLOAD,RMODE(64)
