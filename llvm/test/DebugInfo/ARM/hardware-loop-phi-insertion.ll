; RUN: llc --stop-after=hardware-loops < %s | FileCheck %s

;; Tests that Hardware Loop Insertion does not insert new phi nodes after debug
;; records when they appear immediately after the last existing phi node.

; CHECK-LABEL: for.body:
; CHECK-NEXT: = phi i32
; CHECK-NEXT: = phi i32
; CHECK-NEXT: call void @llvm.dbg.value

source_filename = "repro.c"
target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv8.1m.main-arm-none-eabi"

@z = dso_local local_unnamed_addr global i32 42, align 4, !dbg !0
@arr = dso_local local_unnamed_addr global [10 x i32] zeroinitializer, align 4, !dbg !5

define dso_local void @func1() local_unnamed_addr #0 !dbg !18 {
entry:
  %0 = load i32, ptr @z, align 4, !tbaa !26
  br label %for.body, !dbg !30

for.body:                                         ; preds = %entry, %for.body
  %p1.04 = phi ptr [ @arr, %entry ], [ %incdec.ptr, %for.body ]
  %i.03 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  tail call void @llvm.dbg.value(metadata ptr %p1.04, metadata !23, metadata !DIExpression()), !dbg !25
  store i32 %0, ptr %p1.04, align 4, !dbg !32, !tbaa !26
  %inc = add nuw nsw i32 %i.03, 1, !dbg !34
  %incdec.ptr = getelementptr inbounds i8, ptr %p1.04, i32 4, !dbg !35
  %exitcond.not = icmp eq i32 %inc, 10, !dbg !36
  br i1 %exitcond.not, label %for.end, label %for.body, !dbg !30, !llvm.loop !37

for.end:                                          ; preds = %for.body
  ret void, !dbg !41
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!11, !12, !13, !14, !15, !16}
!llvm.ident = !{!17}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "z", scope: !2, file: !3, line: 2, type: !8, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang version 19.0.0git", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "repro.c", directory: "/home/gbtozers/dev/upstream-llvm")
!4 = !{!0, !5}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = distinct !DIGlobalVariable(name: "arr", scope: !2, file: !3, line: 1, type: !7, isLocal: false, isDefinition: true)
!7 = !DICompositeType(tag: DW_TAG_array_type, baseType: !8, size: 320, elements: !9)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !{!10}
!10 = !DISubrange(count: 10)
!11 = !{i32 7, !"Dwarf Version", i32 5}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{i32 1, !"wchar_size", i32 4}
!14 = !{i32 1, !"min_enum_size", i32 4}
!15 = !{i32 7, !"frame-pointer", i32 2}
!16 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!17 = !{!"clang version 19.0.0git"}
!18 = distinct !DISubprogram(name: "func1", scope: !3, file: !3, line: 4, type: !19, scopeLine: 5, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !21)
!19 = !DISubroutineType(types: !20)
!20 = !{null}
!21 = !{!23}
!22 = !DILocalVariable(name: "i", scope: !18, file: !3, line: 6, type: !8)
!23 = !DILocalVariable(name: "p1", scope: !18, file: !3, line: 7, type: !24)
!24 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 32)
!25 = !DILocation(line: 0, scope: !18)
!26 = !{!27, !27, i64 0}
!27 = !{!"int", !28, i64 0}
!28 = !{!"omnipotent char", !29, i64 0}
!29 = !{!"Simple C/C++ TBAA"}
!30 = !DILocation(line: 8, column: 3, scope: !31)
!31 = distinct !DILexicalBlock(scope: !18, file: !3, line: 8, column: 3)
!32 = !DILocation(line: 9, column: 10, scope: !33)
!33 = distinct !DILexicalBlock(scope: !31, file: !3, line: 8, column: 3)
!34 = !DILocation(line: 8, column: 27, scope: !33)
!35 = !DILocation(line: 8, column: 32, scope: !33)
!36 = !DILocation(line: 8, column: 21, scope: !33)
!37 = distinct !{!37, !30, !38, !39, !40}
!38 = !DILocation(line: 9, column: 12, scope: !31)
!39 = !{!"llvm.loop.mustprogress"}
!40 = !{!"llvm.loop.unroll.disable"}
!41 = !DILocation(line: 10, column: 1, scope: !18)
