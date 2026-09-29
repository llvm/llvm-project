; RUN: opt -passes=lcssa -S %s | FileCheck %s
source_filename = "small.c"

@a = common dso_local global i32 0, align 4, !dbg !0

; CHECK-LABEL: @f
define dso_local void @f() !dbg !10 {
entry:
  %0 = load i32, ptr @a, align 4, !dbg !17
  %tobool1 = icmp eq i32 %0, 0, !dbg !18
  br i1 %tobool1, label %for.end, label %for.inc.lr.ph, !dbg !18

for.inc.lr.ph:                                    ; preds = %entry
  br label %for.inc, !dbg !18

for.inc:                                          ; preds = %for.inc, %for.inc.lr.ph
  %1 = phi i32 [ %0, %for.inc.lr.ph ], [ %inc, %for.inc ]
  call void @llvm.dbg.label(metadata !14), !dbg !19
  %inc = add nsw i32 %1, 1, !dbg !20
  %tobool = icmp eq i32 %inc, 0, !dbg !18
  br i1 %tobool, label %for.cond.for.end_crit_edge, label %for.inc, !dbg !18, !llvm.loop !21

for.cond.for.end_crit_edge:                       ; preds = %for.inc
  store i32 %inc, ptr @a, align 4, !dbg !17
  br label %for.end, !dbg !18

for.end:                                          ; preds = %entry, %for.cond.for.end_crit_edge
  ret void, !dbg !23
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.label(metadata) #0

attributes #0 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "small.c", directory: "./")
!4 = !{}
!5 = !{!0}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = distinct !DISubprogram(name: "f", scope: !3, file: !3, line: 3, type: !11, isLocal: false, isDefinition: true, scopeLine: 4, isOptimized: true, unit: !2, retainedNodes: !13)
!11 = !DISubroutineType(types: !12)
!12 = !{null}
!13 = !{!14}
!14 = !DILabel(scope: !15, name: "L", file: !3, line: 6)
!15 = distinct !DILexicalBlock(scope: !16, file: !3, line: 5, column: 3)
!16 = distinct !DILexicalBlock(scope: !10, file: !3, line: 5, column: 3)
!17 = !DILocation(line: 5, column: 10, scope: !15)
!18 = !DILocation(line: 5, column: 3, scope: !16)
!19 = !DILocation(line: 6, column: 3, scope: !15)
!20 = !DILocation(line: 5, column: 14, scope: !15)
!21 = distinct !{!21, !18, !22}
!22 = !DILocation(line: 6, column: 5, scope: !16)
!23 = !DILocation(line: 7, column: 1, scope: !10)

