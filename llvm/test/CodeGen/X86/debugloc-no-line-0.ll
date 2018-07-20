; RUN: llc -O0 -mtriple=x86_64-unknown-linux-gnu -stop-before="regallocfast" -o - %s | FileCheck %s
;
; We check that all the instructions in bb4 now have a debug-location
; annotation, and that the annotation is identical to the one on e.g.,
; the jmp to bb4.
;
; CHECK: JMP{{.*}}%bb.4, debug-location ![[JUMPLOC:[0-9]+]]
; CHECK: bb.4.entry:
; CHECK: successors:
; CHECK: JE{{.*}}debug-location ![[JUMPLOC]]
; CHECK: JMP{{.*}}debug-location ![[JUMPLOC]]

define i32 @main() !dbg !12 {
entry:
  %add = add nsw i32 undef, 1, !dbg !16
  switch i32 %add, label %sw.epilog [
    i32 1, label %sw.bb
    i32 2, label %sw.bb2
  ], !dbg !17

sw.bb:                                            ; preds = %entry
  br label %sw.epilog, !dbg !20

sw.bb2:                                           ; preds = %entry
  br label %sw.epilog, !dbg !22

sw.epilog:                                        ; preds = %sw.bb2, %sw.bb, %entry
  ret i32 4711, !dbg !23
}
!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!11}

!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, emissionKind: FullDebug, enums: !4)
!3 = !DIFile(filename: "foo.c", directory: ".")
!4 = !{}
!7 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{!"clang"}
!12 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 4, type: !13, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: false, unit: !2, retainedNodes: !4)
!13 = !DISubroutineType(types: !14)
!14 = !{!7}
!16 = !DILocation(line: 6, column: 13, scope: !12)
!17 = !DILocation(line: 6, column: 3, scope: !12)
!19 = distinct !DILexicalBlock(scope: !12, file: !3, line: 7, column: 5)
!20 = !DILocation(line: 10, column: 7, scope: !19)
!22 = !DILocation(line: 13, column: 7, scope: !19)
!23 = !DILocation(line: 24, column: 1, scope: !12)
