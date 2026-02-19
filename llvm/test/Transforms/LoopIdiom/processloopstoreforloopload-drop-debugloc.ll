; RUN: opt < %s -passes=loop-idiom -S | FileCheck %s

; Function Attrs: noinline nounwind uwtable
define dso_local void @fun(ptr noalias noundef %a, ptr noalias noundef %b) #0 !dbg !10 {

; The newly created memcpy should not have the debugloc
; so make sure that the CallInst line ends without "!dbg"
; CHECK-LABEL: entry:
; CHECK: call void @llvm.memcpy.p0.p0.i64{{\(.*\)$}}
entry:
  tail call void @llvm.dbg.value(metadata ptr %a, metadata !17, metadata !DIExpression()), !dbg !18
  tail call void @llvm.dbg.value(metadata ptr %b, metadata !19, metadata !DIExpression()), !dbg !18
  tail call void @llvm.dbg.value(metadata i64 2047, metadata !20, metadata !DIExpression()), !dbg !23
  br label %for.body, !dbg !24

for.body:                                         ; preds = %entry, %for.body
  %i.01 = phi i64 [ 2047, %entry ], [ %dec, %for.body ]
  tail call void @llvm.dbg.value(metadata i64 %i.01, metadata !20, metadata !DIExpression()), !dbg !23
  %arrayidx = getelementptr inbounds i32, ptr %b, i64 %i.01, !dbg !25
  %0 = load i32, ptr %arrayidx, align 4, !dbg !25
  %arrayidx1 = getelementptr inbounds i32, ptr %a, i64 %i.01, !dbg !28
  store i32 %0, ptr %arrayidx1, align 4, !dbg !29
  %dec = add nsw i64 %i.01, -1, !dbg !30
  tail call void @llvm.dbg.value(metadata i64 %dec, metadata !20, metadata !DIExpression()), !dbg !23
  %cmp = icmp sge i64 %dec, 0, !dbg !31
  br i1 %cmp, label %for.body, label %for.end, !dbg !24, !llvm.loop !32

for.end:                                          ; preds = %for.body
  ret void, !dbg !35
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 18.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "1423.c", directory: "/home/linuxbrew/llvm-debug/LoopIdiomRecognize")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 2}
!10 = distinct !DISubprogram(name: "fun", scope: !1, file: !1, line: 1, type: !11, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !16)
!11 = !DISubroutineType(types: !12)
!12 = !{null, !13, !13}
!13 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !14)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!16 = !{}
!17 = !DILocalVariable(name: "a", arg: 1, scope: !10, file: !1, line: 1, type: !13)
!18 = !DILocation(line: 0, scope: !10)
!19 = !DILocalVariable(name: "b", arg: 2, scope: !10, file: !1, line: 1, type: !13)
!20 = !DILocalVariable(name: "i", scope: !21, file: !1, line: 2, type: !22)
!21 = distinct !DILexicalBlock(scope: !10, file: !1, line: 2, column: 5)
!22 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!23 = !DILocation(line: 0, scope: !21)
!24 = !DILocation(line: 2, column: 5, scope: !21)
!25 = !DILocation(line: 3, column: 16, scope: !26)
!26 = distinct !DILexicalBlock(scope: !27, file: !1, line: 2, column: 38)
!27 = distinct !DILexicalBlock(scope: !21, file: !1, line: 2, column: 5)
!28 = !DILocation(line: 3, column: 9, scope: !26)
!29 = !DILocation(line: 3, column: 14, scope: !26)
!30 = !DILocation(line: 2, column: 34, scope: !27)
!31 = !DILocation(line: 2, column: 27, scope: !27)
!32 = distinct !{!32, !24, !33, !34}
!33 = !DILocation(line: 4, column: 5, scope: !21)
!34 = !{!"llvm.loop.mustprogress"}
!35 = !DILocation(line: 5, column: 1, scope: !10)
