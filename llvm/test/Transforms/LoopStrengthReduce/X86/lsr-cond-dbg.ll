; RUN: opt -S -loop-reduce < %s | FileCheck %s

; During Loop Strength Reduce, if the terminating condition for the loop is not
; immediately adjacent to the terminating branch and it has more than one use,
; a clone of the condition will be created just before the terminating branch
; and will be used as the branch condition.
; The purpose of this test is to check that the presence of a debug intrinsic
; between the condition and branch does not trigger this behaviour, as this
; would cause debug info to affect CodeGen.

; CHECK-LABEL: i:
; CHECK-NOT: icmp
; CHECK: [[COND:%.*]] = icmp eq i8
; CHECK-NEXT: call void @llvm.dbg.value(metadata i1 [[COND]]
; CHECK-NEXT: br i1 [[COND]]


target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@b = dso_local local_unnamed_addr global i8 0, align 1, !dbg !0
@a = dso_local local_unnamed_addr global i32 0, align 4, !dbg !6
@c = dso_local local_unnamed_addr global i8 0, align 1, !dbg !9
define dso_local signext i16 @d() local_unnamed_addr !dbg !17 {
entry:
  %0 = load i32, i32* @a, align 4
  %tobool2.not = icmp eq i32 %0, 0
  %1 = load i8, i8* @b, align 1, !dbg !24
  %cmp.not13 = icmp eq i8 %1, 7, !dbg !24
  br i1 %cmp.not13, label %cleanup, label %if.end.preheader, !dbg !24

if.end.preheader:                                 ; preds = %entry
  br label %if.end, !dbg !24

if.end:                                           ; preds = %if.end.preheader, %i
  %2 = phi i8 [ %add, %i ], [ %1, %if.end.preheader ]
  br i1 %tobool2.not, label %i, label %if.end4, !dbg !24

if.end4:                                          ; preds = %if.end
  %conv5 = trunc i32 %0 to i16, !dbg !24
  br label %cleanup, !dbg !24

i:                                                ; preds = %if.end
  %add = add i8 %2, 1, !dbg !24
  store i8 %add, i8* @b, align 1, !dbg !24
  %cmp.not = icmp eq i8 %add, 7, !dbg !24
  call void @llvm.dbg.value(metadata i1 %cmp.not, metadata !23, metadata !DIExpression(DW_OP_LLVM_convert, 1, DW_ATE_unsigned, DW_OP_LLVM_convert, 32, DW_ATE_unsigned, DW_OP_stack_value)), !dbg !24
  br i1 %cmp.not, label %cleanup.loopexit, label %if.end, !dbg !24

cleanup.loopexit:                                 ; preds = %i
  br label %cleanup

cleanup:                                          ; preds = %cleanup.loopexit, %if.end4, %entry
  %cmp.not12 = phi i1 [ %cmp.not13, %if.end4 ], [ %cmp.not13, %entry ], [ %cmp.not, %cleanup.loopexit ]
  %retval.0 = phi i16 [ %conv5, %if.end4 ], [ undef, %entry ], [ undef, %cleanup.loopexit ]
  %3 = load i8, i8* @c, align 1
  %conv8 = sext i8 %3 to i16
  %retval.1 = select i1 %cmp.not12, i16 %conv8, i16 %retval.0
  ret i16 %retval.1, !dbg !24
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!12, !13, !14, !15}
!llvm.ident = !{!16}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "b", scope: !2, file: !3, line: 2, type: !11, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 13.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "lsr-cond-dbg.c", directory: "/")
!4 = !{}
!5 = !{!6, !0, !9}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 1, type: !8, isLocal: false, isDefinition: true)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !DIGlobalVariableExpression(var: !10, expr: !DIExpression())
!10 = distinct !DIGlobalVariable(name: "c", scope: !2, file: !3, line: 2, type: !11, isLocal: false, isDefinition: true)
!11 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!12 = !{i32 7, !"Dwarf Version", i32 4}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{i32 1, !"wchar_size", i32 4}
!15 = !{i32 7, !"uwtable", i32 1}
!16 = !{!"clang version 13.0.0"}
!17 = distinct !DISubprogram(name: "d", scope: !3, file: !3, line: 3, type: !18, scopeLine: 3, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !21)
!18 = !DISubroutineType(types: !19)
!19 = !{!20}
!20 = !DIBasicType(name: "short", size: 16, encoding: DW_ATE_signed)
!21 = !{!23}
!22 = distinct !DILexicalBlock(scope: !17, file: !3, line: 4, column: 3)
!23 = !DILocalVariable(name: "g", scope: !22, file: !3, line: 7, type: !8)
!24 = !DILocation(line: 5, column: 9, scope: !22)

