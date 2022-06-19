; RUN: opt -disable-output "-passes=print<scalar-evolution>" < %s 2>&1 | FileCheck %s

; Make sure poison value tracking works in the presence of @llvm.dbg
; intrinsics.  Unfortunately, I was not able to reduce this file
; further while still keeping the debug info well formed.

define void @foo(i32 %n, i32* %arr) !dbg !7 {
; CHECK-LABEL: Classifying expressions for: @foo
entry:
  %cmp1 = icmp slt i32 0, %n, !dbg !12
  br i1 %cmp1, label %for.body.lr.ph, label %for.end, !dbg !15

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body, !dbg !15

for.body:                                         ; preds = %for.inc, %for.body.lr.ph
  %i.02 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.inc ]
  %add = add nsw i32 %i.02, 50, !dbg !16
  call void @llvm.dbg.value(metadata i32 %add, i64 0, metadata !18, metadata !19), !dbg !20
  %idxprom = sext i32 %add to i64, !dbg !21

; CHECK:  %idxprom = sext i32 %add to i64
; CHECK-NEXT:  -->  {50,+,1}<nuw><nsw><%for.body>

  %arrayidx = getelementptr inbounds i32, i32* %arr, i64 %idxprom, !dbg !21
  store i32 100, i32* %arrayidx, align 4, !dbg !22
  br label %for.inc, !dbg !23

for.inc:                                          ; preds = %for.body
  %inc = add nsw i32 %i.02, 1, !dbg !24
  %cmp = icmp slt i32 %inc, %n, !dbg !12
  br i1 %cmp, label %for.body, label %for.cond.for.end_crit_edge, !dbg !15, !llvm.loop !25

for.cond.for.end_crit_edge:                       ; preds = %for.inc
  br label %for.end, !dbg !15

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry
  ret void, !dbg !27
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.9.0  (llvm/trunk 271857)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "x.c", directory: "/Users/sanjoy/Code/clang/build/debug+asserts-x86")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 2}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"PIC Level", i32 2}
!6 = !{!"clang version 3.9.0  (llvm/trunk 271857)"}
!7 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10, !11}
!10 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64, align: 64)
!12 = !DILocation(line: 2, column: 24, scope: !13)
!13 = distinct !DILexicalBlock(scope: !14, file: !1, line: 2, column: 6)
!14 = distinct !DILexicalBlock(scope: !7, file: !1, line: 2, column: 6)
!15 = !DILocation(line: 2, column: 6, scope: !14)
!16 = !DILocation(line: 3, column: 14, scope: !17)
!17 = distinct !DILexicalBlock(scope: !13, file: !1, line: 2, column: 34)
!18 = !DILocalVariable(name: "k", scope: !17, file: !1, line: 3, type: !10)
!19 = !DIExpression()
!20 = !DILocation(line: 3, column: 8, scope: !17)
!21 = !DILocation(line: 4, column: 4, scope: !17)
!22 = !DILocation(line: 4, column: 11, scope: !17)
!23 = !DILocation(line: 5, column: 6, scope: !17)
!24 = !DILocation(line: 2, column: 30, scope: !13)
!25 = distinct !{!25, !26}
!26 = !DILocation(line: 2, column: 6, scope: !7)
!27 = !DILocation(line: 6, column: 1, scope: !7)
