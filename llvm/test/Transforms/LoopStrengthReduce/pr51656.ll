; RUN: opt -loop-reduce -S %s | FileCheck %s
; RUN: opt --try-experimental-debuginfo-iterators -loop-reduce -S %s | FileCheck %s

;; This test ensures that no attempt is made to translate long SCEVs into
;; DIExpressions. Attempting the translation can use excessive resources and
;; result in crashes.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@c = dso_local local_unnamed_addr global i32 0, align 4, !dbg !0
@b = dso_local local_unnamed_addr global i32 0, align 4, !dbg !9
@a = dso_local local_unnamed_addr global i32 0, align 4, !dbg !5

define dso_local i32 @d() local_unnamed_addr #0 !dbg !16 {
entry:
  %b.promoted = load i32, ptr @b, align 4, !tbaa !29
  %mul = mul nsw i32 %b.promoted, %b.promoted, !dbg !33
  %mul.1 = mul nsw i32 %mul, %mul, !dbg !33
  %mul.2 = mul nsw i32 %mul.1, %mul.1, !dbg !33
  %mul.3 = mul nsw i32 %mul.2, %mul.2, !dbg !33
  %mul.4 = mul nsw i32 %mul.3, %mul.3, !dbg !33
  %mul.5 = mul nsw i32 %mul.4, %mul.4, !dbg !33
  %mul.6 = mul nsw i32 %mul.5, %mul.5, !dbg !33
  %mul.7 = mul nsw i32 %mul.6, %mul.6, !dbg !33
  %mul.8 = mul nsw i32 %mul.7, %mul.7, !dbg !33
  %mul.9 = mul nsw i32 %mul.8, %mul.8, !dbg !33
  store i32 49, ptr @c, align 4, !dbg !36, !tbaa !29
  store i32 %mul.9, ptr @b, align 4, !dbg !37, !tbaa !29
  %.pr = load i32, ptr @a, align 4, !dbg !38, !tbaa !29
  %tobool.not8 = icmp eq i32 %.pr, 0, !dbg !39
  br i1 %tobool.not8, label %for.end3, label %for.body2.preheader, !dbg !39

for.body2.preheader:                              ; preds = %entry
  br label %for.body2, !dbg !39

for.body2:                                        ; preds = %for.body2.preheader, %for.body2
  %0 = phi i32 [ %sub, %for.body2 ], [ %.pr, %for.body2.preheader ]
  %sub = sub nsw i32 %0, %mul.9, !dbg !40
  ; CHECK: call void @llvm.dbg.value(metadata !DIArgList(i32 poison, i32 %mul.9), metadata ![[VAR_e:[0-9]+]], metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_minus, DW_OP_LLVM_convert, 32, DW_ATE_signed, DW_OP_LLVM_convert, 64, DW_ATE_signed, DW_OP_stack_value))
  call void @llvm.dbg.value(metadata i32 %sub, metadata !20, metadata !DIExpression(DW_OP_LLVM_convert, 32, DW_ATE_signed, DW_OP_LLVM_convert, 64, DW_ATE_signed, DW_OP_stack_value)), !dbg !41
  %tobool.not = icmp eq i32 %sub, 0, !dbg !39
  br i1 %tobool.not, label %for.cond1.for.end3_crit_edge, label %for.body2, !dbg !39, !llvm.loop !42

for.cond1.for.end3_crit_edge:                     ; preds = %for.body2
  store i32 0, ptr @a, align 4, !dbg !40, !tbaa !29
  br label %for.end3, !dbg !39

for.end3:                                         ; preds = %for.cond1.for.end3_crit_edge, %entry
  ret i32 poison, !dbg !45
}

declare void @llvm.dbg.value(metadata, metadata, metadata) #1

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!11, !12, !13, !14}
!llvm.ident = !{!15}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "c", scope: !2, file: !7, line: 2, type: !8, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 14.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.c", directory: "/test")
!4 = !{!5, !9, !0}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !7, line: 2, type: !8, isLocal: false, isDefinition: true)
!7 = !DIFile(filename: "./test.c", directory: "/test")
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !DIGlobalVariableExpression(var: !10, expr: !DIExpression())
!10 = distinct !DIGlobalVariable(name: "b", scope: !2, file: !7, line: 2, type: !8, isLocal: false, isDefinition: true)
!11 = !{i32 7, !"Dwarf Version", i32 4}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{i32 1, !"wchar_size", i32 4}
!14 = !{i32 7, !"uwtable", i32 1}
!15 = !{!"clang version 14.0.0"}
!16 = distinct !DISubprogram(name: "d", scope: !7, file: !7, line: 3, type: !17, scopeLine: 3, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !19)
!17 = !DISubroutineType(types: !18)
!18 = !{!8}
!19 = !{!20}
; CHECK: ![[VAR_e]] = !DILocalVariable(name: "e", scope: !21, file: !7, line: 8, type: !24)
!20 = !DILocalVariable(name: "e", scope: !21, file: !7, line: 8, type: !24)
!21 = distinct !DILexicalBlock(scope: !22, file: !7, line: 7, column: 14)
!22 = distinct !DILexicalBlock(scope: !23, file: !7, line: 7, column: 3)
!23 = distinct !DILexicalBlock(scope: !16, file: !7, line: 7, column: 3)
!24 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint64_t", file: !25, line: 27, baseType: !26)
!25 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/stdint-uintn.h", directory: "")
!26 = !DIDerivedType(tag: DW_TAG_typedef, name: "__uint64_t", file: !27, line: 44, baseType: !28)
!27 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/types.h", directory: "")
!28 = !DIBasicType(name: "long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!29 = !{!30, !30, i64 0}
!30 = !{!"int", !31, i64 0}
!31 = !{!"omnipotent char", !32, i64 0}
!32 = !{!"Simple C/C++ TBAA"}
!33 = !DILocation(line: 6, column: 7, scope: !34)
!34 = distinct !DILexicalBlock(scope: !35, file: !7, line: 5, column: 3)
!35 = distinct !DILexicalBlock(scope: !16, file: !7, line: 5, column: 3)
!36 = !DILocation(line: 0, scope: !16)
!37 = !DILocation(line: 0, scope: !34)
!38 = !DILocation(line: 7, column: 10, scope: !22)
!39 = !DILocation(line: 7, column: 3, scope: !23)
!40 = !DILocation(line: 8, column: 20, scope: !21)
!41 = !DILocation(line: 0, scope: !21)
!42 = distinct !{!42, !39, !43, !44}
!43 = !DILocation(line: 9, column: 3, scope: !23)
!44 = !{!"llvm.loop.mustprogress"}
!45 = !DILocation(line: 10, column: 1, scope: !16)
