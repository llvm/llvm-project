; RUN: opt %s -loop-deletion -S | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

@a = common local_unnamed_addr global i32 0, align 4, !dbg !0

define i32 @b() local_unnamed_addr !dbg !12 {
entry:
  call void @llvm.dbg.value(metadata i32 0, metadata !16, metadata !DIExpression()), !dbg !17
  br label %for.cond, !dbg !18

for.cond:                                         ; preds = %for.cond, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.cond ], !dbg !20
  call void @llvm.dbg.value(metadata i32 %i.0, metadata !16, metadata !DIExpression()), !dbg !17
  %inc = add nuw nsw i32 %i.0, 1, !dbg !21
  call void @llvm.dbg.value(metadata i32 %inc, metadata !16, metadata !DIExpression()), !dbg !17
  %exitcond = icmp ne i32 %inc, 3, !dbg !23
  br i1 %exitcond, label %for.cond, label %for.end, !dbg !24, !llvm.loop !25

; CHECK: call void @llvm.dbg.value(metadata i32 undef, metadata !16, metadata !DIExpression()), !dbg !17
; CHECK-NEXT: %call = tail call i32 {{.*}} @patatino()
for.end:                                          ; preds = %for.cond
  %call = tail call i32 (...) @patatino() #3, !dbg !27
  %0 = load i32, i32* @a, align 4, !dbg !28
  ret i32 %0, !dbg !33
}

declare i32 @patatino(...) local_unnamed_addr

define i32 @main() local_unnamed_addr !dbg !34 {
entry:
  %call = call i32 @b(), !dbg !35
  ret i32 0, !dbg !36
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9, !10}
!llvm.ident = !{!11}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 8.0.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: GNU)
!3 = !DIFile(filename: "a.c", directory: "/Users/davide/work/llvm-project-20170507/build-debug/bin")
!4 = !{}
!5 = !{!0}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{i32 7, !"PIC Level", i32 2}
!11 = !{!"clang version 8.0.0 "}
!12 = distinct !DISubprogram(name: "b", scope: !3, file: !3, line: 2, type: !13, scopeLine: 2, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !15)
!13 = !DISubroutineType(types: !14)
!14 = !{!6}
!15 = !{!16}
!16 = !DILocalVariable(name: "i", scope: !12, file: !3, line: 3, type: !6)
!17 = !DILocation(line: 3, column: 7, scope: !12)
!18 = !DILocation(line: 4, column: 8, scope: !19)
!19 = distinct !DILexicalBlock(scope: !12, file: !3, line: 4, column: 3)
!20 = !DILocation(line: 0, scope: !19)
!21 = !DILocation(line: 4, column: 23, scope: !22)
!22 = distinct !DILexicalBlock(scope: !19, file: !3, line: 4, column: 3)
!23 = !DILocation(line: 4, column: 17, scope: !22)
!24 = !DILocation(line: 4, column: 3, scope: !19)
!25 = distinct !{!25, !24, !26}
!26 = !DILocation(line: 5, column: 5, scope: !19)
!27 = !DILocation(line: 6, column: 3, scope: !12)
!28 = !DILocation(line: 7, column: 10, scope: !12)
!33 = !DILocation(line: 7, column: 3, scope: !12)
!34 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 9, type: !13, scopeLine: 9, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!35 = !DILocation(line: 9, column: 14, scope: !34)
!36 = !DILocation(line: 9, column: 19, scope: !34)
