; RUN: opt < %s -loop-reduce -S | FileCheck %s
; RUN: opt --try-experimental-debuginfo-iterators < %s -loop-reduce -S | FileCheck %s

; Test that LSR does not produce invalid debug info when a debug value is
; salvaged during LSR by adding additional location operands, then becomes
; undef, and finally recovered by SCEV salvaging.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define i32 @_Z3fooiii(i32 %Result, i32 %Step, i32 %Last) local_unnamed_addr !dbg !7 {
; CHECK-LABEL: @_Z3fooiii(
entry:
  call void @llvm.dbg.value(metadata i32 %Result, metadata !12, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 %Step, metadata !13, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 %Last, metadata !14, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 0, metadata !16, metadata !DIExpression()), !dbg !17
  br label %do.body, !dbg !18

do.body:                                          ; preds = %do.body, %entry
; CHECK-LABEL: do.body: 
  %Result.addr.0 = phi i32 [ %Result, %entry ], [ %or, %do.body ]
  %Itr.0 = phi i32 [ 0, %entry ], [ %add, %do.body ], !dbg !17
; CHECK-NOT: call void @llvm.dbg.value(metadata !DIArgList
; CHECK: call void @llvm.dbg.value(metadata !DIArgList(i32 %lsr.iv, i32 %Step), metadata ![[VAR_ITR:[0-9]+]], metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_minus, DW_OP_LLVM_arg, 1, DW_OP_div, DW_OP_LLVM_arg, 1, DW_OP_mul, DW_OP_stack_value))
  call void @llvm.dbg.value(metadata i32 %Itr.0, metadata !16, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 %Result.addr.0, metadata !12, metadata !DIExpression()), !dbg !17
  %add = add nsw i32 %Itr.0, %Step, !dbg !19
  call void @llvm.dbg.value(metadata i32 %add, metadata !16, metadata !DIExpression()), !dbg !17
  %call = tail call i32 @_Z3barv(), !dbg !21
  call void @llvm.dbg.value(metadata i32 %call, metadata !15, metadata !DIExpression()), !dbg !17
  %shl = shl i32 %call, %add, !dbg !22
  %or = or i32 %shl, %Result.addr.0, !dbg !23
  call void @llvm.dbg.value(metadata i32 %or, metadata !12, metadata !DIExpression()), !dbg !17
  %and = and i32 %call, %Last, !dbg !24
  %tobool.not = icmp eq i32 %and, 0, !dbg !25
  br i1 %tobool.not, label %do.end, label %do.body, !dbg !26, !llvm.loop !27

do.end:                                           ; preds = %do.body
  ret i32 %or, !dbg !30
}

declare !dbg !31 dso_local i32 @_Z3barv() local_unnamed_addr

declare void @llvm.dbg.value(metadata, metadata, metadata)

; CHECK: ![[VAR_ITR]] = !DILocalVariable(name: "Itr"

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 13.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 13.0.0"}
!7 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooiii", scope: !1, file: !1, line: 3, type: !8, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !10, !10, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12, !13, !14, !15, !16}
!12 = !DILocalVariable(name: "Result", arg: 1, scope: !7, file: !1, line: 3, type: !10)
!13 = !DILocalVariable(name: "Step", arg: 2, scope: !7, file: !1, line: 3, type: !10)
!14 = !DILocalVariable(name: "Last", arg: 3, scope: !7, file: !1, line: 3, type: !10)
!15 = !DILocalVariable(name: "Bar", scope: !7, file: !1, line: 4, type: !10)
!16 = !DILocalVariable(name: "Itr", scope: !7, file: !1, line: 5, type: !10)
!17 = !DILocation(line: 0, scope: !7)
!18 = !DILocation(line: 6, column: 3, scope: !7)
!19 = !DILocation(line: 7, column: 9, scope: !20)
!20 = distinct !DILexicalBlock(scope: !7, file: !1, line: 6, column: 6)
!21 = !DILocation(line: 8, column: 11, scope: !20)
!22 = !DILocation(line: 9, column: 20, scope: !20)
!23 = !DILocation(line: 9, column: 12, scope: !20)
!24 = !DILocation(line: 10, column: 16, scope: !7)
!25 = !DILocation(line: 10, column: 12, scope: !7)
!26 = !DILocation(line: 10, column: 3, scope: !20)
!27 = distinct !{!27, !18, !28, !29}
!28 = !DILocation(line: 10, column: 22, scope: !7)
!29 = !{!"llvm.loop.mustprogress"}
!30 = !DILocation(line: 11, column: 3, scope: !7)
!31 = !DISubprogram(name: "bar", linkageName: "_Z3barv", scope: !1, file: !1, line: 1, type: !32, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!32 = !DISubroutineType(types: !33)
!33 = !{!10}
