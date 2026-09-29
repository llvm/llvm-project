;; Test that we get a parser error when a debug intrinsic appears in the same
;; module as a debug record.
; RUN: not llvm-as < %s 2>&1 | FileCheck %s
; ModuleID = '<stdin>'
source_filename = "<stdin>"

define dso_local i32 @f(i32 %a) !dbg !7 {
entry:
    #dbg_value(!DIArgList(i32 %a), !12, !DIExpression(), !14)
; CHECK: <stdin>:[[@LINE+1]]:8: error: llvm.dbg intrinsic should not appear in a module using non-intrinsic debug info
  call void @llvm.dbg.value(metadata i32 %a, metadata !12, metadata !DIExpression()), !dbg !14
  ret i32 %a, !dbg !18
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 18.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "print.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 18.0.0"}
!7 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 3, type: !8, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12, !13}
!12 = !DILocalVariable(name: "a", arg: 1, scope: !7, file: !1, line: 3, type: !10)
!13 = !DILocalVariable(name: "b", scope: !7, file: !1, line: 3, type: !10)
!14 = !DILocation(line: 3, column: 15, scope: !7)
!15 = distinct !DIAssignID()
!16 = !DILocation(line: 3, column: 20, scope: !7)
!17 = !DILocation(line: 3, column: 25, scope: !7)
!18 = !DILocation(line: 3, column: 30, scope: !7)
