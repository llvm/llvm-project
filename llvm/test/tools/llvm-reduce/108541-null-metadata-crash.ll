; RUN: llvm-reduce %s -o %t --delta-passes=di-metadata --test FileCheck --test-arg %s --test-arg --input-file --abort-on-invalid-reduction
; CHECK: , !dbg !11

;; Tests for the bug fixed in PR#108541, where the presence of null metadata
;; could result in a crash.

define i1 @ham() {
bb:
  %call = call fastcc i32 @hoge()
  ret i1 false
}

define fastcc i32 @hoge() {
bb:
  br i1 poison, label %bb1, label %bb2

bb1:                                              ; preds = %bb
  br i1 false, label %bb2, label %bb2, !dbg !11

bb2:                                              ; preds = %bb1, %bb1, %bb
  ret i32 0
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 20.0.0git", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !2, globals: !2, imports: !3, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "108541-metadata-crash.cpp", directory: "/tmp")
!2 = !{}
!3 = !{!4}
!4 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !6, file: !7, line: 134)
!5 = !DINamespace(name: "std", scope: null)
!6 = !DISubprogram(name: "abort", scope: !7, file: !7, line: 730, type: !8, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: DISPFlagOptimized)
!7 = !DIFile(filename: "108541-metadata-crash.cpp", directory: "")
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !DILocation(line: 26, column: 11, scope: !12)
!12 = distinct !DILexicalBlock(scope: !14, file: !13, line: 26, column: 11)
!13 = !DIFile(filename: "108541-metadata-crash.cpp", directory: "/tmp")
!14 = distinct !DILexicalBlock(scope: !15, file: !13, line: 25, column: 5)
!15 = distinct !DILexicalBlock(scope: !16, file: !13, line: 24, column: 9)
!16 = distinct !DILexicalBlock(scope: !17, file: !13, line: 14, column: 3)
!17 = distinct !DILexicalBlock(scope: !18, file: !13, line: 13, column: 3)
!18 = distinct !DILexicalBlock(scope: !19, file: !13, line: 13, column: 3)
!19 = distinct !DISubprogram(name: "hoge", linkageName: "hoge", scope: !13, file: !13, line: 10, type: !20, scopeLine: 11, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!20 = distinct !DISubroutineType(types: !2)
