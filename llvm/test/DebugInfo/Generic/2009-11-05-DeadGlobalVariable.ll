; RUN: llc %s -o /dev/null
; Here variable bar is optimized away. Do not trip over while trying to generate debug info.

source_filename = "test/DebugInfo/Generic/2009-11-05-DeadGlobalVariable.ll"

; Function Attrs: nounwind readnone ssp uwtable
define i32 @foo() #0 !dbg !6 {
entry:
  ret i32 42, !dbg !11
}

attributes #0 = { nounwind readnone ssp uwtable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.0 (trunk 139632)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !3)
!1 = !DIFile(filename: "fb.c", directory: "/private/tmp")
!2 = !{}
!3 = !{!4}
!4 = !DIGlobalVariableExpression(var: !5, expr: !DIExpression())
!5 = !DIGlobalVariable(name: "bar", scope: !6, file: !1, line: 2, type: !9, isLocal: true, isDefinition: true)
!6 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !7, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: true, unit: !0)
!7 = !DISubroutineType(types: !8)
!8 = !{!9}
!9 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !{i32 1, !"Debug Info Version", i32 3}
!11 = !DILocation(line: 3, column: 3, scope: !12)
!12 = distinct !DILexicalBlock(scope: !6, file: !1, line: 1, column: 11)

