; RUN: opt -loop-unroll -unroll-runtime -unroll-allow-remainder -unroll-count=4 -unroll-remainder -S %s -o - | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@b = common local_unnamed_addr global i32 0, align 4, !dbg !0
@a = common local_unnamed_addr global i32* null, align 8, !dbg !6

; Test that loop remainder unrolling doesn't corrupt debuginfo. This example
; used to cause an assert, but also test that the unrolled backwards branches
; have the same DILocation.

; CHECK-LABEL: func_c
; CHECK-LABEL: for.body.lr.ph:
; CHECK: br i1 %[[CMP0:.*]], label %[[PRE:.*]], label %for.body.prol.loopexit, !dbg !24
; CHECK-LABEL: for.body:
; CHECK: br i1 %[[CMP1:.*]], label %[[CRIT_EDGE:.*]], label %for.body, !dbg !24, !llvm.loop !30
; CHECK-LABEL: for.cond.for.end_crit_edge:
; CHECK: br label %for.end, !dbg !24
; CHECK-LABEL: for.body.prol.1:
; CHECK: br i1 %[[CMP2:.*]], label %for.body.prol.2, label %[[EXIT:.*]], !dbg !24
; CHECK-LABEL: for.body.prol.2:
define i32 @func_c() local_unnamed_addr #0 !dbg !14 {
entry:
  %.pr = load i32, i32* @b, align 4, !dbg !17, !tbaa !20
  %tobool1 = icmp eq i32 %.pr, 0, !dbg !24
  br i1 %tobool1, label %for.end, label %for.body.lr.ph, !dbg !24

for.body.lr.ph:
  %a.promoted = load i32*, i32** @a, align 8, !dbg !25, !tbaa !26
  %0 = sub i32 -2, %.pr, !dbg !24
  %1 = and i32 %0, -2, !dbg !24
  %2 = add i32 %.pr, %1, !dbg !24
  br label %for.body, !dbg !24

for.body:
  %3 = phi i32* [ %a.promoted, %for.body.lr.ph ], [ %6, %for.body ], !dbg !28
  %4 = phi i32 [ %.pr, %for.body.lr.ph ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %3, i64 1, !dbg !28
  %5 = load i32, i32* %arrayidx, align 4, !dbg !28, !tbaa !20
  %conv = sext i32 %5 to i64, !dbg !28
  %6 = inttoptr i64 %conv to i32*, !dbg !28
  %add = add nsw i32 %4, 2, !dbg !29
  %tobool = icmp eq i32 %add, 0, !dbg !24
  br i1 %tobool, label %for.cond.for.end_crit_edge, label %for.body, !dbg !24, !llvm.loop !30

for.cond.for.end_crit_edge:
  %7 = add i32 %2, 2, !dbg !24
  store i32* %6, i32** @a, align 8, !dbg !25, !tbaa !26
  store i32 %7, i32* @b, align 4, !dbg !32, !tbaa !20
  br label %for.end, !dbg !24

for.end:
  ret i32 undef, !dbg !33
}

; CHECK-LABEL: func_d
define void @func_d() local_unnamed_addr #1 !dbg !34 {
entry:
  ret void, !dbg !37
}

attributes #0 = { norecurse nounwind uwtable }
attributes #0 = { norecurse nounwind readnone uwtable }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!10, !11, !12}
!llvm.ident = !{!13}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "b", scope: !2, file: !3, line: 2, type: !9, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 6.0.0 (http://llvm.org/git/clang.git 044091b728654e62444a7ea10e6efb489c705bed) (http://llvm.org/git/llvm.git 1c7b55afdb1c5791e0557d9e32e2dd07c7acb2b0)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "loop.c", directory: "/work/projects/src/tests/unroll-debug-info")
!4 = !{}
!5 = !{!6, !0}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 1, type: !8, isLocal: false, isDefinition: true)
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{!"clang version 6.0.0 (http://llvm.org/git/clang.git 044091b728654e62444a7ea10e6efb489c705bed) (http://llvm.org/git/llvm.git 1c7b55afdb1c5791e0557d9e32e2dd07c7acb2b0)"}
!14 = distinct !DISubprogram(name: "c", scope: !3, file: !3, line: 3, type: !15, isLocal: false, isDefinition: true, scopeLine: 3, isOptimized: true, unit: !2, retainedNodes: !4)
!15 = !DISubroutineType(types: !16)
!16 = !{!9}
!17 = !DILocation(line: 4, column: 12, scope: !18)
!18 = distinct !DILexicalBlock(scope: !19, file: !3, line: 4, column: 5)
!19 = distinct !DILexicalBlock(scope: !14, file: !3, line: 4, column: 5)
!20 = !{!21, !21, i64 0}
!21 = !{!"int", !22, i64 0}
!22 = !{!"omnipotent char", !23, i64 0}
!23 = !{!"Simple C/C++ TBAA"}
!24 = !DILocation(line: 4, column: 5, scope: !19)
!25 = !DILocation(line: 5, column: 13, scope: !18)
!26 = !{!27, !27, i64 0}
!27 = !{!"any pointer", !22, i64 0}
!28 = !DILocation(line: 5, column: 15, scope: !18)
!29 = !DILocation(line: 4, column: 21, scope: !18)
!30 = distinct !{!30, !24, !31}
!31 = !DILocation(line: 5, column: 18, scope: !19)
!32 = !DILocation(line: 4, column: 17, scope: !18)
!33 = !DILocation(line: 6, column: 1, scope: !14)
!34 = distinct !DISubprogram(name: "d", scope: !3, file: !3, line: 7, type: !35, isLocal: false, isDefinition: true, scopeLine: 7, isOptimized: true, unit: !2, retainedNodes: !4)
!35 = !DISubroutineType(types: !36)
!36 = !{null}
!37 = !DILocation(line: 7, column: 11, scope: !34)
