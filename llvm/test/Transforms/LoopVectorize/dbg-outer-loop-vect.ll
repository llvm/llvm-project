; RUN: opt < %s -S -passes=loop-vectorize -enable-vplan-native-path -force-vector-interleave=1 -force-vector-width=4 | FileCheck %s

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"

define void @foo(ptr %h) !dbg !4 {
entry:
  call void @llvm.dbg.value(metadata ptr %h, metadata !10, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.value(metadata i64 0, metadata !11, metadata !DIExpression()), !dbg !21
  br label %for.cond1.preheader, !dbg !22

for.cond1.preheader:                              ; preds = %entry, %for.cond.cleanup3
  %i.023 = phi i64 [ 0, %entry ], [ %inc13, %for.cond.cleanup3 ]
  call void @llvm.dbg.value(metadata i64 %i.023, metadata !11, metadata !DIExpression()), !dbg !21
  call void @llvm.dbg.value(metadata i64 0, metadata !14, metadata !DIExpression()), !dbg !23
  br label %for.cond5.preheader, !dbg !24

for.cond5.preheader:                              ; preds = %for.cond1.preheader, %for.cond5.preheader
  %l.022 = phi i64 [ 0, %for.cond1.preheader ], [ %inc10, %for.cond5.preheader ]
  call void @llvm.dbg.value(metadata i64 %l.022, metadata !14, metadata !DIExpression()), !dbg !23
  %0 = getelementptr i32, ptr %h, i64 %l.022
  call void @llvm.dbg.value(metadata i64 0, metadata !17, metadata !DIExpression()), !dbg !26
  store i32 0, ptr %0, align 4, !dbg !27
  call void @llvm.dbg.value(metadata i64 1, metadata !17, metadata !DIExpression()), !dbg !26
  %arrayidx.1 = getelementptr i32, ptr %0, i64 1, !dbg !29
  store i32 1, ptr %arrayidx.1, align 4, !dbg !27
  call void @llvm.dbg.value(metadata i64 2, metadata !17, metadata !DIExpression()), !dbg !26
  %arrayidx.2 = getelementptr i32, ptr %0, i64 2, !dbg !29
  store i32 2, ptr %arrayidx.2, align 4, !dbg !27
  call void @llvm.dbg.value(metadata i64 3, metadata !17, metadata !DIExpression()), !dbg !26
  %arrayidx.3 = getelementptr i32, ptr %0, i64 3, !dbg !29
  store i32 3, ptr %arrayidx.3, align 4, !dbg !27
  call void @llvm.dbg.value(metadata i64 4, metadata !17, metadata !DIExpression()), !dbg !26
  %inc10 = add nuw nsw i64 %l.022, 1, !dbg !30
  call void @llvm.dbg.value(metadata i64 %inc10, metadata !14, metadata !DIExpression()), !dbg !23
  %exitcond.not = icmp eq i64 %inc10, 5, !dbg !31
  br i1 %exitcond.not, label %for.cond.cleanup3, label %for.cond5.preheader, !dbg !24, !llvm.loop !32

for.cond.cleanup3:                                ; preds = %for.cond5.preheader
  %inc13 = add nuw nsw i64 %i.023, 1, !dbg !34
  call void @llvm.dbg.value(metadata i64 %inc13, metadata !11, metadata !DIExpression()), !dbg !21
  %exitcond24.not = icmp eq i64 %inc13, 23, !dbg !35
  br i1 %exitcond24.not, label %exit, label %for.cond1.preheader, !dbg !22, !llvm.loop !36

exit:                                 ; preds = %for.cond.cleanup3
  ret void, !dbg !25
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "outer-loop-vect.c", directory: "/test/file/path")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!4 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 8, type: !5, scopeLine: 8, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !9)
!5 = !DISubroutineType(types: !6)
!6 = !{null, !7}
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !{!10, !11, !14, !17}
!10 = !DILocalVariable(name: "h", arg: 1, scope: !4, file: !1, line: 8, type: !7)
!11 = !DILocalVariable(name: "i", scope: !12, file: !1, line: 10, type: !13)
!12 = distinct !DILexicalBlock(scope: !4, file: !1, line: 10, column: 3)
!13 = !DIBasicType(name: "unsigned long", size: 64, encoding: DW_ATE_unsigned)
!14 = !DILocalVariable(name: "l", scope: !15, file: !1, line: 11, type: !13)
!15 = distinct !DILexicalBlock(scope: !16, file: !1, line: 11, column: 5)
!16 = distinct !DILexicalBlock(scope: !12, file: !1, line: 10, column: 3)
!17 = !DILocalVariable(name: "j", scope: !18, file: !1, line: 12, type: !13)
!18 = distinct !DILexicalBlock(scope: !19, file: !1, line: 12, column: 7)
!19 = distinct !DILexicalBlock(scope: !15, file: !1, line: 11, column: 5)
!20 = !DILocation(line: 0, scope: !4)
!21 = !DILocation(line: 0, scope: !12)
!22 = !DILocation(line: 10, column: 3, scope: !12)
!23 = !DILocation(line: 0, scope: !15)
!24 = !DILocation(line: 11, column: 5, scope: !15)
!25 = !DILocation(line: 14, column: 1, scope: !4)
!26 = !DILocation(line: 0, scope: !18)
!27 = !DILocation(line: 13, column: 11, scope: !28)
!28 = distinct !DILexicalBlock(scope: !18, file: !1, line: 12, column: 7)
!29 = !DILocation(line: 13, column: 2, scope: !28)
!30 = !DILocation(line: 11, column: 32, scope: !19)
!31 = !DILocation(line: 11, column: 26, scope: !19)
!32 = distinct !{!32, !24, !33}
!33 = !DILocation(line: 13, column: 13, scope: !15)
!34 = !DILocation(line: 10, column: 30, scope: !16)
!35 = !DILocation(line: 10, column: 24, scope: !16)
!36 = distinct !{!36, !22, !37, !38}
!37 = !DILocation(line: 13, column: 13, scope: !12)
!38 = !{!"llvm.loop.vectorize.enable", i1 true}
