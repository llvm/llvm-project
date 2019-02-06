; RUN: llc -start-after=codegenprepare -stop-before expand-isel-pseudos -o - %s | FileCheck %s

; Test that dbg.values of an SSA variable that's not used in a basic block,
; is converted to a DBG_VALUE in that same basic block. We know that %1 is
; live from the entry bb to the exit bb, is allocated a vreg because it's
; used across basic blocks. We should be able to produce a DBG_VALUE that
; refers to it in the nextbb block.

source_filename = "debug.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define dso_local i32 @main(i32 %arg0, i32 %arg1) local_unnamed_addr !dbg !11 {
entry:
  %0 = add i32 %arg0, 42, !dbg !26
  %1 = add i32 %arg1, 101, !dbg !26
  %cmp = icmp eq i32 %1, 0
  br i1 %cmp, label %nextbb, label %exit, !dbg !26

nextbb:
; CHECK-LABEL: bb.{{.*}}.nextbb
  %2 = mul i32 %0, %arg1, !dbg !26
; CHECK: IMUL32rr
  call void @llvm.dbg.value(metadata i32 %1, metadata !16, metadata !DIExpression()), !dbg !27
; CHECK-NEXT: DBG_VALUE
  br label %exit, !dbg !26

; CHECK-LABEL: bb.{{.*}}.exit
exit:
  %3 = phi i32 [ 12, %entry ], [ %2, %nextbb ], !dbg !26
  %4 = add i32 %1, %3, !dbg !26
  ret i32 %4, !dbg !26
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9}
!llvm.ident = !{!10}

!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (x)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4)
!3 = !DIFile(filename: "debug.c", directory: "")
!4 = !{}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{!"clang version 7.0.0 (x)"}
!11 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 3, type: !12, isLocal: false, isDefinition: true, scopeLine: 3, isOptimized: true, unit: !2, retainedNodes: !14)
!12 = !DISubroutineType(types: !13)
!13 = !{!6}
!14 = !{!16}
!16 = !DILocalVariable(name: "y", scope: !11, file: !3, line: 5, type: !6)
!26 = !DILocation(line: 4, column: 7, scope: !11)
!27 = !DILocation(line: 5, column: 7, scope: !11)
!31 = !{!32, !32, i64 0}
!32 = !{!"int", !33, i64 0}
!33 = !{!"omnipotent char", !34, i64 0}
!34 = !{!"Simple C/C++ TBAA"}
