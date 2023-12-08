; RUN: opt -passes=strip-dead-debug-info -disable-output < %s
source_filename = "test/Transforms/StripSymbols/2010-08-25-crash.ll"

; Function Attrs: nounwind ssp
define i32 @foo() #0 !dbg !9 {
entry:
  ret i32 0, !dbg !12
}

attributes #0 = { nounwind ssp }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 2.8 (trunk 112062)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !3)
!1 = !DIFile(filename: "/tmp/a.c", directory: "/Volumes/Lalgate/clean/D.CW")
!2 = !{}
!3 = !{!4}
!4 = !DIGlobalVariableExpression(var: !5, expr: !DIExpression())
!5 = !DIGlobalVariable(name: "i", linkageName: "i", scope: !1, file: !1, line: 2, type: !6, isLocal: true, isDefinition: true)
!6 = !DIDerivedType(tag: DW_TAG_const_type, scope: !1, file: !1, baseType: !7)
!7 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !{i32 1, !"Debug Info Version", i32 3}
!9 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: !1, file: !1, line: 3, type: !10, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, unit: !0)
!10 = !DISubroutineType(types: !11)
!11 = !{!7}
!12 = !DILocation(line: 3, column: 13, scope: !13)
!13 = distinct !DILexicalBlock(scope: !9, file: !1, line: 3, column: 11)

