; Make sure we don't crash when writing bitcode.
; RUN: opt < %s -passes=loop-deletion -o /dev/null
; RUN: opt < %s -passes=loop-deletion -o /dev/null --try-experimental-debuginfo-iterators

define void @f(i1 %arg) {
  br label %bb1

bb1:                                              ; preds = %bb1, %0
  call void @llvm.dbg.value(metadata i16 undef, metadata !1, metadata !DIExpression()), !dbg !11
  br i1 %arg, label %bb1, label %bb3

bb3:                                              ; preds = %bb1
  ret void
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!9}
!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !DILocalVariable(name: "i", scope: !2, file: !3, line: 31, type: !7)
!2 = distinct !DILexicalBlock(scope: !4, file: !3, line: 31, column: 9)
!3 = !DIFile(filename: "foo.c", directory: "/bar")
!4 = distinct !DISubprogram(name: "f", scope: !3, file: !3, line: 26, type: !5, scopeLine: 27, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !9, retainedNodes: !10)
!5 = !DISubroutineType(types: !6)
!6 = !{!7, !8, !7}
!7 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
!8 = !DIBasicType(size: 16, encoding: DW_ATE_signed)
!9 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "My Compiler", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !10, retainedTypes: !10, globals: !10)
!10 = !{}
!11 = !DILocation(line: 31, column: 13, scope: !2)
