; Check for setting dbg.value undef which depends on instruction which is going to be dead
; RUN: opt -bdce %s -S | FileCheck %s

; Generated from this
; char a, b;
; void optimize_me_not();
; int main() {
;  char l_177 = 2;
;  for (; b <= 0; b++)
;    for (; a >= 0; a--) {
;      ++l_177;
;      optimize_me_not();
;    }
; }

@b = common dso_local local_unnamed_addr global i8 0, align 1, !dbg !0
@a = common dso_local local_unnamed_addr global i8 0, align 1, !dbg !6

define dso_local i32 @main() local_unnamed_addr !dbg !13 {
entry:
;CHECK: call void @llvm.dbg.value(metadata i8 2
;CHECK: call void @llvm.dbg.value(metadata i8 2
  call void @llvm.dbg.value(metadata i8 2, metadata !17, metadata !DIExpression()), !dbg !18
  %.pr = load i8, i8* @b, align 1, !dbg !19
  call void @llvm.dbg.value(metadata i8 2, metadata !17, metadata !DIExpression()), !dbg !18
  %cmp5 = icmp slt i8 %.pr, 1, !dbg !22
  br i1 %cmp5, label %for.cond2thread-pre-split.preheader, label %for.end9, !dbg !23

for.cond2thread-pre-split.preheader:              ; preds = %entry
  br label %for.cond2thread-pre-split, !dbg !23
for.cond2thread-pre-split:                        ; preds = %for.cond2thread-pre-split.preheader, %for.inc7
;CHECK: call void @llvm.dbg.value(metadata i8 undef
  %l_177.06 = phi i8 [ %l_177.1.lcssa, %for.inc7 ], [ 2, %for.cond2thread-pre-split.preheader ]
  call void @llvm.dbg.value(metadata i8 %l_177.06, metadata !17, metadata !DIExpression()), !dbg !18
;CHECK: call void @llvm.dbg.value(metadata i8 undef
  %.pr1 = load i8, i8* @a, align 1, !dbg !24
  call void @llvm.dbg.value(metadata i8 %l_177.06, metadata !17, metadata !DIExpression()), !dbg !18
  %cmp42 = icmp sgt i8 %.pr1, -1, !dbg !27
  br i1 %cmp42, label %for.body6.preheader, label %for.inc7, !dbg !28

for.body6.preheader:                              ; preds = %for.cond2thread-pre-split
  br label %for.body6, !dbg !28

for.body6:                                        ; preds = %for.body6.preheader, %for.body6
;CHECK: call void @llvm.dbg.value(metadata i8 undef
;CHECK: call void @llvm.dbg.value(metadata i8 undef
  %l_177.13 = phi i8 [ %inc, %for.body6 ], [ %l_177.06, %for.body6.preheader ]
  call void @llvm.dbg.value(metadata i8 %l_177.13, metadata !17, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata i8 %l_177.13, metadata !17, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !18
  tail call void (...) @optimize_me_not(),  !dbg !29
  %inc = add i8 %l_177.13, 1, !dbg !31
  %0 = load i8, i8* @a, align 1, !dbg !32
  %dec = add i8 %0, -1, !dbg !32
  store i8 %dec, i8* @a, align 1, !dbg !32
  call void @llvm.dbg.value(metadata i8 %inc, metadata !17, metadata !DIExpression()), !dbg !18
  %cmp4 = icmp sgt i8 %dec, -1, !dbg !27
  br i1 %cmp4, label %for.body6, label %for.inc7.loopexit, !dbg !28, !llvm.loop !33

for.inc7.loopexit:                                ; preds = %for.body6
  %inc.lcssa = phi i8 [ %inc, %for.body6 ], !dbg !31
  br label %for.inc7, !dbg !35

for.inc7:                                         ; preds = %for.inc7.loopexit, %for.cond2thread-pre-split
;CHECK: call void @llvm.dbg.value(metadata i8 undef
  %l_177.1.lcssa = phi i8 [ %l_177.06, %for.cond2thread-pre-split ], [ %inc.lcssa, %for.inc7.loopexit ], !dbg !18
  %1 = load i8, i8* @b, align 1, !dbg !35
  %inc8 = add i8 %1, 1, !dbg !35
  store i8 %inc8, i8* @b, align 1, !dbg !35
  call void @llvm.dbg.value(metadata i8 %l_177.1.lcssa, metadata !17, metadata !DIExpression()), !dbg !18
  %cmp = icmp slt i8 %inc8, 1, !dbg !22
  br i1 %cmp, label %for.cond2thread-pre-split, label %for.end9.loopexit, !dbg !23, !llvm.loop !36

for.end9.loopexit:                                ; preds = %for.inc7
  br label %for.end9, !dbg !38

for.end9:                                         ; preds = %for.end9.loopexit, %entry
  ret i32 0, !dbg !38
}

declare dso_local void @optimize_me_not(...) local_unnamed_addr 

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!9, !10, !11}
!llvm.ident = !{!12}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "b", scope: !2, file: !3, line: 3, type: !8, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 10.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None)
!3 = !DIFile(filename: "test", directory: "test")
!4 = !{}
!5 = !{!6, !0}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 3, type: !8, isLocal: false, isDefinition: true)
!8 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!9 = !{i32 2, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{i32 1, !"wchar_size", i32 4}
!12 = !{!"clang version 10.0.0"}
!13 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 4, type: !14, scopeLine: 4, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!14 = !DISubroutineType(types: !15)
!15 = !{!16}
!16 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!17 = !DILocalVariable(name: "l_177", scope: !13, file: !3, line: 5, type: !8)
!18 = !DILocation(line: 0, scope: !13)
!19 = !DILocation(line: 6, column: 10, scope: !20)
!20 = distinct !DILexicalBlock(scope: !21, file: !3, line: 6, column: 3)
!21 = distinct !DILexicalBlock(scope: !13, file: !3, line: 6, column: 3)
!22 = !DILocation(line: 6, column: 12, scope: !20)
!23 = !DILocation(line: 6, column: 3, scope: !21)
!24 = !DILocation(line: 7, column: 12, scope: !25)
!25 = distinct !DILexicalBlock(scope: !26, file: !3, line: 7, column: 5)
!26 = distinct !DILexicalBlock(scope: !20, file: !3, line: 7, column: 5)
!27 = !DILocation(line: 7, column: 14, scope: !25)
!28 = !DILocation(line: 7, column: 5, scope: !26)
!29 = !DILocation(line: 9, column: 7, scope: !30)
!30 = distinct !DILexicalBlock(scope: !25, file: !3, line: 7, column: 25)
!31 = !DILocation(line: 8, column: 7, scope: !30)
!32 = !DILocation(line: 7, column: 21, scope: !25)
!33 = distinct !{!33, !28, !34}
!34 = !DILocation(line: 10, column: 5, scope: !26)
!35 = !DILocation(line: 6, column: 19, scope: !20)
!36 = distinct !{!36, !23, !37}
!37 = !DILocation(line: 10, column: 5, scope: !21)
!38 = !DILocation(line: 11, column: 1, scope: !13)
