; RUN: llc %s -o - -print-after=machine-cse -mtriple=x86_64-- 2>&1 | FileCheck %s --match-full-lines

; CHECK: %5:gr32 = SUB32ri8 %0:gr32(tied-def 0), 1, implicit-def $eflags, debug-location !24; a.c:3:13
; CHECK-NEXT: %10:gr32 = MOVSX32rr8 %4:gr8
; CHECK-NEXT: JCC_1 %bb.2, 15, implicit $eflags, debug-location !25; a.c:3:18

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

@a = dso_local local_unnamed_addr global i32 0, align 4, !dbg !0

; Function Attrs: norecurse nounwind readonly ssp uwtable
define dso_local i32 @b(i8 signext %0) local_unnamed_addr #0 !dbg !12 {
  call void @llvm.dbg.value(metadata i8 %0, metadata !17, metadata !DIExpression()), !dbg !18
  %2 = load i32, i32* @a, align 4, !dbg !19, !tbaa !20
  %3 = icmp sgt i32 %2, 1, !dbg !24
  br i1 %3, label %8, label %4, !dbg !25

4:                                                ; preds = %1
  %5 = sext i8 %0 to i32, !dbg !26
  %6 = ashr i32 %5, %2, !dbg !27
  %7 = icmp eq i32 %6, 0, !dbg !27
  br i1 %7, label %10, label %8, !dbg !28

8:                                                ; preds = %4, %1
  %9 = sext i8 %0 to i32, !dbg !29
  br label %10, !dbg !28

10:                                               ; preds = %4, %8
  %11 = phi i32 [ %9, %8 ], [ 0, %4 ], !dbg !28
  ret i32 %11, !dbg !30
}

define dso_local i32 @main() local_unnamed_addr #0 !dbg !31 {
  %1 = call i32 @b(i8 signext 0), !dbg !34
  ret i32 %1, !dbg !35
}

declare void @llvm.dbg.value(metadata, metadata, metadata) #1

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9, !10}
!llvm.ident = !{!11}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 11.0.0 (https://github.com/llvm/llvm-project 75cfd382201978615cca1c91c2d9f14f8b7af56d)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None, sysroot: "/")
!3 = !DIFile(filename: "a.c", directory: "/Users/davide/work/build/bin")
!4 = !{}
!5 = !{!0}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 7, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{i32 7, !"PIC Level", i32 2}
!11 = !{!"clang version 11.0.0 (https://github.com/llvm/llvm-project 75cfd382201978615cca1c91c2d9f14f8b7af56d)"}
!12 = distinct !DISubprogram(name: "b", scope: !3, file: !3, line: 2, type: !13, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !16)
!13 = !DISubroutineType(types: !14)
!14 = !{!6, !15}
!15 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!16 = !{!17}
!17 = !DILocalVariable(name: "c", arg: 1, scope: !12, file: !3, line: 2, type: !15)
!18 = !DILocation(line: 0, scope: !12)
!19 = !DILocation(line: 3, column: 11, scope: !12)
!20 = !{!21, !21, i64 0}
!21 = !{!"int", !22, i64 0}
!22 = !{!"omnipotent char", !23, i64 0}
!23 = !{!"Simple C/C++ TBAA"}
!24 = !DILocation(line: 3, column: 13, scope: !12)
!25 = !DILocation(line: 3, column: 18, scope: !12)
!26 = !DILocation(line: 3, column: 21, scope: !12)
!27 = !DILocation(line: 3, column: 23, scope: !12)
!28 = !DILocation(line: 3, column: 10, scope: !12)
!29 = !DILocation(line: 4, column: 16, scope: !12)
!30 = !DILocation(line: 3, column: 3, scope: !12)
!31 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 8, type: !32, scopeLine: 9, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!32 = !DISubroutineType(types: !33)
!33 = !{!6}
!34 = !DILocation(line: 10, column: 10, scope: !31)
!35 = !DILocation(line: 10, column: 3, scope: !31)
