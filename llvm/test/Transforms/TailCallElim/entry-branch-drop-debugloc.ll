; RUN: opt -S -passes=tailcallelim < %s | FileCheck %s

define dso_local i32 @func(i32 noundef %a) !dbg !10 {
; CHECK: entry
; CHECK: br label %tailrecurse{{$}}
entry:
  tail call void @llvm.dbg.value(metadata i32 %a, metadata !15, metadata !DIExpression()), !dbg !16
  %cmp = icmp sgt i32 %a, 1, !dbg !17
  br i1 %cmp, label %if.then, label %if.end, !dbg !19

if.then:                                          ; preds = %entry
  %sub = sub nsw i32 %a, 1, !dbg !20
  %call = call i32 @func(i32 noundef %sub), !dbg !22
  %mul = mul nsw i32 %a, %call, !dbg !23
  br label %return, !dbg !24

if.end:                                           ; preds = %entry
  br label %return, !dbg !25

return:                                           ; preds = %if.end, %if.then
  %retval.0 = phi i32 [ %mul, %if.then ], [ 1, %if.end ], !dbg !16
  ret i32 %retval.0, !dbg !26
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 19.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "main.c", directory: "/root/llvm-test/TailCallElim")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 2}
!10 = distinct !DISubprogram(name: "func", scope: !1, file: !1, line: 1, type: !11, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !14)
!11 = !DISubroutineType(types: !12)
!12 = !{!13, !13}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !{}
!15 = !DILocalVariable(name: "a", arg: 1, scope: !10, file: !1, line: 1, type: !13)
!16 = !DILocation(line: 0, scope: !10)
!17 = !DILocation(line: 2, column: 11, scope: !18)
!18 = distinct !DILexicalBlock(scope: !10, file: !1, line: 2, column: 9)
!19 = !DILocation(line: 2, column: 9, scope: !10)
!20 = !DILocation(line: 3, column: 27, scope: !21)
!21 = distinct !DILexicalBlock(scope: !18, file: !1, line: 2, column: 16)
!22 = !DILocation(line: 3, column: 20, scope: !21)
!23 = !DILocation(line: 3, column: 18, scope: !21)
!24 = !DILocation(line: 3, column: 9, scope: !21)
!25 = !DILocation(line: 5, column: 5, scope: !10)
!26 = !DILocation(line: 6, column: 1, scope: !10)
