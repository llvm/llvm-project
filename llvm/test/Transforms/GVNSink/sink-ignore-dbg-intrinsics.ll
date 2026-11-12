; RUN: opt < %s -passes=gvn-sink -S | FileCheck %s

; Test that GVNSink correctly performs the sink optimization in the presence of debug information
; Test that GVNSink correctly merges the debug locations of sinked instruction, eg, propagating
; the merged debug location of `%add` and `%add1` to the sinked add instruction.

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @fun(i32 noundef %a, i32 noundef %b) #0 !dbg !10 {
; CHECK-LABEL: define dso_local i32 @fun(
; CHECK-SAME:    i32 noundef [[A:%.*]], i32 noundef [[B:%.*]])
; CHECK:       if.end:
; CHECK:         [[B_SINK:%.*]] = phi i32 [ [[B]], %if.else ], [ [[A]], %if.then ]
; CHECK:         [[ADD1:%.*]] = add nsw i32 [[B_SINK]], 1, !dbg [[DBG:![0-9]+]]
; CHECK:         [[XOR2:%.*]] = xor i32 [[ADD1]], 1, !dbg [[DBG:![0-9]+]]
; CHECK:       [[DBG]] = !DILocation(line: 0,
;
entry:
  tail call void @llvm.dbg.value(metadata i32 %a, metadata !15, metadata !DIExpression()), !dbg !16
  tail call void @llvm.dbg.value(metadata i32 %b, metadata !17, metadata !DIExpression()), !dbg !16
  %cmp = icmp sgt i32 %b, 10, !dbg !18
  br i1 %cmp, label %if.then, label %if.else, !dbg !20

if.then:                                          ; preds = %entry
  %add = add nsw i32 %a, 1, !dbg !21
  tail call void @llvm.dbg.value(metadata i32 %add, metadata !23, metadata !DIExpression()), !dbg !24
  %xor = xor i32 %add, 1, !dbg !25
  tail call void @llvm.dbg.value(metadata i32 %xor, metadata !26, metadata !DIExpression()), !dbg !24
  tail call void @llvm.dbg.value(metadata i32 %xor, metadata !27, metadata !DIExpression()), !dbg !16
  br label %if.end, !dbg !28

if.else:                                          ; preds = %entry
  %add1 = add nsw i32 %b, 1, !dbg !29
  tail call void @llvm.dbg.value(metadata i32 %add1, metadata !31, metadata !DIExpression()), !dbg !32
  %xor2 = xor i32 %add1, 1, !dbg !33
  tail call void @llvm.dbg.value(metadata i32 %xor2, metadata !34, metadata !DIExpression()), !dbg !32
  tail call void @llvm.dbg.value(metadata i32 %xor2, metadata !27, metadata !DIExpression()), !dbg !16
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %ret.0 = phi i32 [ %xor, %if.then ], [ %xor2, %if.else ], !dbg !35
  tail call void @llvm.dbg.value(metadata i32 %ret.0, metadata !27, metadata !DIExpression()), !dbg !16
  ret i32 %ret.0, !dbg !36
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { noinline nounwind uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 18.0.0git (https://github.com/llvm/llvm-project.git 5dfcb3e5d1d16bb4f8fce52b3c089119ed977e7f)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "main.c", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 2}
!10 = distinct !DISubprogram(name: "fun", scope: !1, file: !1, line: 1, type: !11, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !14)
!11 = !DISubroutineType(types: !12)
!12 = !{!13, !13, !13}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !{}
!15 = !DILocalVariable(name: "a", arg: 1, scope: !10, file: !1, line: 1, type: !13)
!16 = !DILocation(line: 0, scope: !10)
!17 = !DILocalVariable(name: "b", arg: 2, scope: !10, file: !1, line: 1, type: !13)
!18 = !DILocation(line: 3, column: 11, scope: !19)
!19 = distinct !DILexicalBlock(scope: !10, file: !1, line: 3, column: 9)
!20 = !DILocation(line: 3, column: 9, scope: !10)
!21 = !DILocation(line: 4, column: 20, scope: !22)
!22 = distinct !DILexicalBlock(scope: !19, file: !1, line: 3, column: 17)
!23 = !DILocalVariable(name: "a1", scope: !22, file: !1, line: 4, type: !13)
!24 = !DILocation(line: 0, scope: !22)
!25 = !DILocation(line: 5, column: 21, scope: !22)
!26 = !DILocalVariable(name: "a2", scope: !22, file: !1, line: 5, type: !13)
!27 = !DILocalVariable(name: "ret", scope: !10, file: !1, line: 2, type: !13)
!28 = !DILocation(line: 7, column: 5, scope: !22)
!29 = !DILocation(line: 8, column: 20, scope: !30)
!30 = distinct !DILexicalBlock(scope: !19, file: !1, line: 7, column: 12)
!31 = !DILocalVariable(name: "b1", scope: !30, file: !1, line: 8, type: !13)
!32 = !DILocation(line: 0, scope: !30)
!33 = !DILocation(line: 9, column: 21, scope: !30)
!34 = !DILocalVariable(name: "b2", scope: !30, file: !1, line: 9, type: !13)
!35 = !DILocation(line: 0, scope: !19)
!36 = !DILocation(line: 12, column: 5, scope: !10)
