; RUN: opt -S %s -passes=loop-deletion | FileCheck %s

;; Generated from this C source:
;; static int f(int p) { return p * p * 2; }
;; static int zero() { return 0; }
;; void fun() {
;;   for (int __attribute__((nodebug)) i = zero(); i < 0; ++i) {
;;     f(i);
;;     f(i + 1);
;;   }
;; }
;;
;; Check that loop-deletion doesn't accidently mistake debug intrinsics for
;; different inlined instances of a variable as the same variable.

; CHECK-LABEL: for.cond.cleanup: ; preds = %entry
; CHECK-NEXT:    #dbg_value({{.+}}, ![[P:[0-9]+]],{{.+}},  ![[DBG1:[0-9]+]]
; CHECK-NEXT:    #dbg_value({{.+}}, ![[P]],       {{.+}},  ![[DBG2:[0-9]+]]

; CHECK-DAG: ![[P]] = !DILocalVariable(name: "p",
; CHECK-DAG: ![[DBG1]] = !DILocation({{.+}}, inlinedAt: ![[IA1:[0-9]+]])
; CHECK-DAG: ![[DBG2]] = !DILocation({{.+}}, inlinedAt: ![[IA2:[0-9]+]])
; CHECK-DAG: ![[IA1]] = distinct !DILocation(line: 5,
; CHECK-DAG: ![[IA2]] = distinct !DILocation(line: 6,

define dso_local void @fun() !dbg !9 {
entry:
  br label %for.cond, !dbg !13

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ], !dbg !15
  %cmp = icmp slt i32 %i.0, 0, !dbg !16
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !dbg !18

for.cond.cleanup:                                 ; preds = %for.cond
  br label %for.end

for.body:                                         ; preds = %for.cond
  call void @llvm.dbg.value(metadata i32 %i.0, metadata !19, metadata !DIExpression()), !dbg !25
  %mul.i = mul nsw i32 %i.0, %i.0, !dbg !28
  %mul1.i = mul nsw i32 %mul.i, 2, !dbg !29
  %add = add nsw i32 %i.0, 1, !dbg !30
  call void @llvm.dbg.value(metadata i32 %add, metadata !19, metadata !DIExpression()), !dbg !31
  %mul.i1 = mul nsw i32 %add, %add, !dbg !33
  %mul1.i2 = mul nsw i32 %mul.i1, 2, !dbg !34
  br label %for.inc, !dbg !35

for.inc:                                          ; preds = %for.body
  %inc = add nsw i32 %i.0, 1, !dbg !36
  br label %for.cond, !dbg !37, !llvm.loop !38

for.end:                                          ; preds = %for.cond.cleanup
  ret void, !dbg !41
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 16.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{!"clang version 16.0.0"}
!9 = distinct !DISubprogram(name: "fun", scope: !1, file: !1, line: 3, type: !10, scopeLine: 3, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !12)
!10 = !DISubroutineType(types: !11)
!11 = !{null}
!12 = !{}
!13 = !DILocation(line: 4, column: 8, scope: !14)
!14 = distinct !DILexicalBlock(scope: !9, file: !1, line: 4, column: 3)
!15 = !DILocation(line: 4, scope: !14)
!16 = !DILocation(line: 4, column: 51, scope: !17)
!17 = distinct !DILexicalBlock(scope: !14, file: !1, line: 4, column: 3)
!18 = !DILocation(line: 4, column: 3, scope: !14)
!19 = !DILocalVariable(name: "p", arg: 1, scope: !20, file: !1, line: 1, type: !23)
!20 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !21, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !24)
!21 = !DISubroutineType(types: !22)
!22 = !{!23, !23}
!23 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!24 = !{!19}
!25 = !DILocation(line: 0, scope: !20, inlinedAt: !26)
!26 = distinct !DILocation(line: 5, column: 5, scope: !27)
!27 = distinct !DILexicalBlock(scope: !17, file: !1, line: 4, column: 61)
!28 = !DILocation(line: 1, column: 32, scope: !20, inlinedAt: !26)
!29 = !DILocation(line: 1, column: 36, scope: !20, inlinedAt: !26)
!30 = !DILocation(line: 6, column: 8, scope: !27)
!31 = !DILocation(line: 0, scope: !20, inlinedAt: !32)
!32 = distinct !DILocation(line: 6, column: 5, scope: !27)
!33 = !DILocation(line: 1, column: 32, scope: !20, inlinedAt: !32)
!34 = !DILocation(line: 1, column: 36, scope: !20, inlinedAt: !32)
!35 = !DILocation(line: 7, column: 3, scope: !27)
!36 = !DILocation(line: 4, column: 56, scope: !17)
!37 = !DILocation(line: 4, column: 3, scope: !17)
!38 = distinct !{!38, !18, !39, !40}
!39 = !DILocation(line: 7, column: 3, scope: !14)
!40 = !{!"llvm.loop.mustprogress"}
!41 = !DILocation(line: 8, column: 1, scope: !9)
